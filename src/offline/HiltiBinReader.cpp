#include "HiltiBinReader.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <charconv>
#include <chrono>

namespace cgmli { namespace offline {
namespace {
[[noreturn]] void fail(const std::string& text) { throw std::runtime_error("HILTBIN1: "+text); }
template<class T> T scalar(const uint8_t* data,bool bigEndian) {
    static_assert(std::is_arithmetic<T>::value,"scalar must be arithmetic");
    std::array<uint8_t,sizeof(T)> bytes;
    const uint16_t one=1;
    const bool hostBig=*reinterpret_cast<const uint8_t*>(&one)==0;
    if(hostBig==bigEndian) std::copy_n(data,sizeof(T),bytes.data());
    else std::reverse_copy(data,data+sizeof(T),bytes.data());
    T value; std::memcpy(&value,bytes.data(),sizeof(T)); return value;
}
void readExact(std::ifstream& file,void* bytes,size_t count) {
    if(count>size_t(std::numeric_limits<std::streamsize>::max())) fail("read exceeds stream size");
    if(count && !file.read(static_cast<char*>(bytes),std::streamsize(count))) fail("truncated record");
}
void seek(std::ifstream& file,uint64_t offset) {
    if(offset>uint64_t(std::numeric_limits<std::streamoff>::max())) fail("file offset is not representable");
    file.clear(); file.seekg(std::streamoff(offset),std::ios::beg);
    if(!file) fail("seek failed");
}
std::vector<std::string> csvFields(const std::string& line) {
    std::vector<std::string> fields; std::string field; bool quoted=false;
    for(size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if(c=='"') {
            if(quoted && i+1<line.size() && line[i+1]=='"') { field+='"'; ++i; }
            else quoted=!quoted;
        } else if(c==',' && !quoted) { fields.push_back(std::move(field)); field.clear(); }
        else field+=c;
    }
    if(quoted) fail("unterminated CSV field"); fields.push_back(std::move(field)); return fields;
}
template<class T> T csvInteger(const std::string& field) {
    T value{}; const auto parsed=std::from_chars(field.data(),field.data()+field.size(),value);
    if(parsed.ec!=std::errc() || parsed.ptr!=field.data()+field.size()) fail("invalid CSV integer");
    return value;
}
uint32_t integer(const YAML::Node& node,const char* key) {
    if(!node[key] || !node[key].IsScalar()) fail(std::string("missing integer ")+key);
    const auto value=node[key].as<int64_t>();
    if(value<0 || uint64_t(value)>std::numeric_limits<uint32_t>::max()) fail(std::string("invalid integer ")+key);
    return uint32_t(value);
}
std::string string(const YAML::Node& node,const char* key) {
    if(!node[key] || !node[key].IsScalar()) fail(std::string("missing string ")+key);
    return node[key].as<std::string>();
}
bool boolean(const YAML::Node& node,const char* key) {
    // The converter writes JSON booleans for clouds and integer 0/1 for Images.
    const auto value=string(node,key);
    if(value=="true" || value=="1") return true;
    if(value=="false" || value=="0") return false;
    fail(std::string("invalid boolean ")+key);
}
RecordIndex header(const std::array<uint8_t,32>& bytes,uint64_t offset) {
    RecordIndex result; result.offset=offset;
    const auto kind=scalar<uint32_t>(bytes.data(),false);
    if(kind<1 || kind>5) fail("unknown record kind at "+std::to_string(offset));
    result.kind=RecordKind(kind); result.metadataBytes=scalar<uint32_t>(bytes.data()+4,false);
    result.sensorNs=scalar<int64_t>(bytes.data()+8,false); result.bagNs=scalar<int64_t>(bytes.data()+16,false);
    result.payloadBytes=scalar<uint64_t>(bytes.data()+24,false); return result;
}
size_t fieldBytes(uint32_t datatype) {
    switch(datatype) {
    case 1: case 2: return 1;
    case 3: case 4: return 2;
    case 5: case 6: case 7: return 4;
    case 8: return 8;
    default: fail("unsupported PointField datatype");
    }
}
double pointScalar(const uint8_t* point,const CloudField& field,bool bigEndian) {
    const auto data=point+field.offset;
    switch(field.datatype) {
    case 1: return scalar<int8_t>(data,bigEndian);
    case 2: return scalar<uint8_t>(data,bigEndian);
    case 3: return scalar<int16_t>(data,bigEndian);
    case 4: return scalar<uint16_t>(data,bigEndian);
    case 5: return scalar<int32_t>(data,bigEndian);
    case 6: return scalar<uint32_t>(data,bigEndian);
    case 7: return scalar<float>(data,bigEndian);
    case 8: return scalar<double>(data,bigEndian);
    default: fail("unsupported point scalar");
    }
}
int64_t pointTime(double seconds) {
    // Robot timestamp is an absolute FLOAT64 second count. Never reduce this to
    // float32 or cast seconds to an integer before multiplying by 1e9.
    const long double ns=static_cast<long double>(seconds)*1000000000.0L;
    if(!std::isfinite(seconds) || seconds<0 || ns>=static_cast<long double>(std::numeric_limits<int64_t>::max()))
        fail("invalid absolute LiDAR point time");
    return int64_t(std::round(ns));
}
}

struct HiltiBinReader::RawRecord { RecordIndex index; YAML::Node metadata; std::vector<uint8_t> payload; };

HiltiBinReader::HiltiBinReader(const std::filesystem::path& path,const HiltiTopics& topics,const ReadLimits& limits)
    : path_(path),topics_(topics),limits_(limits) {
    if(path.extension()==".partial") fail("conversion is incomplete; select finalized data.bin");
    if(!limits.maximumMetadataBytes || !limits.maximumPayloadBytes || !limits.maximumRecords
        || !limits.maximumImagePixels || !limits.maximumScanPoints) fail("invalid read limits");
    std::set<std::string> selected(topics.cameras.begin(),topics.cameras.end());
    if(selected.size()!=4 || selected.count("") || topics.imu.empty() || topics.lidar.empty()
        || selected.count(topics.imu) || selected.count(topics.lidar) || topics.imu==topics.lidar) fail("topics must identify four distinct cameras, an IMU and a LiDAR");
    const bool directory=std::filesystem::is_directory(path_);
    const auto receipt=(directory?path_:path_.parent_path())/"summary.json";
    YAML::Node summary;
    if(std::filesystem::exists(receipt)) {
        std::ifstream file(receipt); summary=YAML::Load(file);
        if(!summary["complete"] || !summary["complete"].as<bool>()) fail("summary.json marks conversion incomplete");
    }
    if(directory) {
        if(!summary.IsMap() || string(summary,"format")!="HILTBIN1" || integer(summary,"version")!=1
            || !summary["topic_files"].IsMap()) fail("directory requires completed per-topic summary.json");
        selected.insert(topics.imu); selected.insert(topics.lidar);
        const auto root=std::filesystem::canonical(path_);
        for(const auto& topic:selected) {
            if(!summary["topic_files"][topic]) fail("summary is missing selected topic: "+topic);
            const auto filename=std::filesystem::u8path(summary["topic_files"][topic].as<std::string>());
            // The converter emits flat filenames. Do not follow a manifest path
            // outside its dataset, including an unexpected symlink destination.
            if(filename.empty() || filename!=filename.filename() || filename.extension()!=".bin") fail("invalid topic filename");
            const auto target=std::filesystem::canonical(root/filename);
            if(target.parent_path()!=root) fail("topic file escapes dataset directory");
            for(const auto& previous:files_) if(std::filesystem::equivalent(previous,target)) fail("two selected topics map to the same file");
            files_.push_back(target); fileTopics_.push_back(topic);
        }
    } else { files_.push_back(path_); fileTopics_.push_back(""); }
    for(const auto& file:files_) {
        inputs_.emplace_back(file,std::ios::binary);
        if(!inputs_.back()) fail("cannot open "+file.u8string());
        fileSizes_.push_back(std::filesystem::file_size(file));
    }
    index();
    if(directory) for(size_t i=0;i<files_.size();++i) {
        const auto stats=summary["topic_stats"][fileTopics_[i]];
        if(stats && ((stats["file_bytes"] && stats["file_bytes"].as<uint64_t>()!=std::filesystem::file_size(files_[i]))
            || (stats["records"] && stats["records"].as<uint64_t>()!=statistics_.topicRecords.at(fileTopics_[i]))))
            fail("topic file differs from completed conversion receipt: "+fileTopics_[i]);
    }
}
bool HiltiBinReader::indexCsv() {
    if(!std::filesystem::is_directory(path_)) return false;
    const auto path=path_/"index.csv"; std::ifstream csv(path,std::ios::binary); if(!csv) return false;
    // The converter's completed manifest is checked by the constructor. An index
    // older than a changed binary is not reused; per-record headers/metadata are
    // still validated when consumed, so the CSV never replaces binary validation.
    const auto modified=std::filesystem::last_write_time(path);
    std::vector<uint64_t> lengths,ends(files_.size(),12);
    IndexStatistics statistics; std::vector<RecordIndex> records;
    std::map<std::string,size_t> byTopic;
    for(size_t f=0;f<files_.size();++f) {
        if(std::filesystem::last_write_time(files_[f])>modified) {
            LOG(INFO)<<"[Dataset] index.csv is older than "<<files_[f].filename().u8string()<<"; rebuilding from binaries";
            return false;
        }
        const auto bytes=std::filesystem::file_size(files_[f]); if(bytes<12) fail("missing binary header");
        seek(inputs_[f],0); std::array<uint8_t,12> magic; readExact(inputs_[f],magic.data(),magic.size());
        if(std::memcmp(magic.data(),"HILTBIN1",8)!=0 || scalar<uint32_t>(magic.data()+8,false)!=1) fail("unsupported magic/version");
        lengths.push_back(bytes); statistics.fileBytes+=bytes; byTopic.emplace(fileTopics_[f],f);
    }
    std::string line; std::getline(csv,line); if(!line.empty() && line.back()=='\r') line.pop_back();
    if(line!="bag_time_ns,sensor_time_ns,kind,connection_id,topic,file,offset,record_bytes") fail("unsupported index.csv schema");
    while(std::getline(csv,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const auto fields=csvFields(line); if(fields.size()!=8) fail("invalid index.csv row");
        const auto selected=byTopic.find(fields[4]); if(selected==byTopic.end()) continue;
        const size_t fileId=selected->second;
        if(fields[5]!=files_[fileId].filename().u8string()) fail("CSV topic/file disagrees with manifest");
        RecordIndex item; item.fileId=fileId; item.bagNs=csvInteger<int64_t>(fields[0]);
        item.sensorNs=csvInteger<int64_t>(fields[1]); const auto kind=csvInteger<uint32_t>(fields[2]);
        csvInteger<uint32_t>(fields[3]); item.offset=csvInteger<uint64_t>(fields[6]);
        item.recordBytes=csvInteger<uint64_t>(fields[7]); item.deferredHeader=true;
        if(kind<1 || kind>4 || item.sensorNs<0 || item.offset!=ends[fileId] || item.offset>lengths[fileId]
            || item.recordBytes<32 || item.recordBytes>lengths[fileId]-item.offset
            || item.recordBytes-32>limits_.maximumMetadataBytes+limits_.maximumPayloadBytes) fail("invalid CSV record bounds/order");
        item.kind=RecordKind(kind); const auto camera=std::find(topics_.cameras.begin(),topics_.cameras.end(),fields[4]);
        if(camera!=topics_.cameras.end()) {
            if(item.kind!=RecordKind::Image && item.kind!=RecordKind::CompressedImage) fail("CSV camera kind mismatch");
            item.camera=int(camera-topics_.cameras.begin());
        } else if(item.kind!=(fields[4]==topics_.imu?RecordKind::Imu:RecordKind::PointCloud2)) fail("CSV sensor kind mismatch");
        ends[fileId]+=item.recordBytes; records.push_back(item);
        ++statistics.topicRecords[fields[4]]; if(++statistics.totalRecords>limits_.maximumRecords) fail("CSV record limit exceeded");
    }
    if(!csv.eof()) fail("CSV index read failed");
    for(size_t f=0;f<files_.size();++f) if(ends[f]!=lengths[f]) fail("CSV does not cover complete selected binary");
    statistics.reusedCsv=true; records_=std::move(records); statistics_=std::move(statistics);
    LOG(INFO)<<"[Dataset] Reused converter index.csv; selected_records="<<records_.size()<<"; binary metadata validation deferred to record reads";
    return true;
}
void HiltiBinReader::index() {
    const auto started=std::chrono::steady_clock::now();
    LOG(INFO) << "[Dataset] Building sensor index; files=" << files_.size();
    bool reused=false;
    try { reused=indexCsv(); }
    catch(const std::exception& error) { LOG(WARNING)<<"[Dataset] CSV index unavailable; scanning binaries: "<<error.what(); }
    if(!reused) for(size_t fileId=0;fileId<files_.size();++fileId) {
        auto& input=inputs_[fileId]; const auto fileBytes=std::filesystem::file_size(files_[fileId]);
        LOG(INFO) << "[Dataset] Indexing " << files_[fileId].u8string() << "; bytes=" << fileBytes;
        statistics_.fileBytes+=fileBytes;
        if(fileBytes<12) fail("missing file header: "+files_[fileId].u8string());
        seek(input,0);
        std::array<uint8_t,12> bytes; readExact(input,bytes.data(),bytes.size());
        if(std::memcmp(bytes.data(),"HILTBIN1",8)!=0 || scalar<uint32_t>(bytes.data()+8,false)!=1) fail("unsupported magic/version");
        for(uint64_t offset=12;offset<fileBytes;) {
            if(fileBytes-offset<32) fail("truncated record header");
            seek(input,offset); std::array<uint8_t,32> raw; readExact(input,raw.data(),raw.size());
            auto item=header(raw,offset); item.fileId=fileId;
            const auto remaining=fileBytes-offset-32;
            if(item.metadataBytes>limits_.maximumMetadataBytes || item.payloadBytes>limits_.maximumPayloadBytes
                || item.metadataBytes>remaining || item.payloadBytes>remaining-item.metadataBytes) fail("oversized or truncated record at "+std::to_string(offset));
            std::string json(item.metadataBytes,'\0'); readExact(input,json.data(),json.size());
            const auto metadata=YAML::Load(json);
            if(!metadata.IsMap()) fail("record metadata must be a JSON object");
            const auto topic=string(metadata,"topic");
            if(!fileTopics_[fileId].empty() && topic!=fileTopics_[fileId]) fail("record topic differs from summary file mapping");
            const auto type=string(metadata,"msgtype"); string(metadata,"frame_id"); integer(metadata,"connection_id");
            ++statistics_.totalRecords; ++statistics_.topicRecords[topic];
            if(statistics_.totalRecords>limits_.maximumRecords) fail("record count exceeds configured memory bound");
            const auto camera=std::find(topics_.cameras.begin(),topics_.cameras.end(),topic);
            const bool selected=camera!=topics_.cameras.end() || topic==topics_.imu || topic==topics_.lidar;
            if(selected) {
                if(item.sensorNs<0) fail("selected sensor record has no header timestamp: "+topic);
                const char* expectedType=item.kind==RecordKind::Image?"sensor_msgs/msg/Image":
                    item.kind==RecordKind::CompressedImage?"sensor_msgs/msg/CompressedImage":
                    item.kind==RecordKind::Imu?"sensor_msgs/msg/Imu":"sensor_msgs/msg/PointCloud2";
                if(type!=expectedType) fail("selected metadata message type disagrees with record kind");
                if(camera!=topics_.cameras.end()) {
                    if(item.kind!=RecordKind::Image && item.kind!=RecordKind::CompressedImage) fail("camera topic has wrong record kind");
                    item.camera=int(camera-topics_.cameras.begin());
                } else if(topic==topics_.imu) {
                    if(item.kind!=RecordKind::Imu || item.payloadBytes!=37*8) fail("IMU record must contain 37 little-endian doubles");
                } else if(item.kind!=RecordKind::PointCloud2) fail("LiDAR topic has wrong record kind");
                records_.push_back(item);
            }
            offset+=32+item.metadataBytes+item.payloadBytes;
        }
    }
    // Bag order is not header-time order. Keep random-access offsets, and sort
    // selected indices once without rewriting or duplicating the large container.
    std::sort(records_.begin(),records_.end(),[](const auto& a,const auto& b) {
        if(a.sensorNs!=b.sensorNs) return a.sensorNs<b.sensorNs;
        if(a.fileId!=b.fileId) return a.fileId<b.fileId;
        return a.offset<b.offset;
    });
    for(size_t i=0;i<records_.size();++i) {
        const auto& item=records_[i];
        auto& stream=item.camera>=0?cameras_[size_t(item.camera)]:(item.kind==RecordKind::Imu?imu_:lidar_);
        if(!stream.empty() && records_[stream.back()].sensorNs>=item.sensorNs) fail("duplicate/non-increasing timestamp in a selected topic");
        stream.push_back(i);
    }
    for(size_t c=0;c<4;++c) if(cameras_[c].empty()) fail("selected camera topic is absent: "+topics_.cameras[c]);
    if(imu_.empty()) fail("selected IMU topic is absent: "+topics_.imu);
    if(lidar_.empty()) fail("selected LiDAR topic is absent: "+topics_.lidar);
    statistics_.selectedRecords=records_.size();
    LOG(INFO) << "[Dataset] Sensor index ready; total_records=" << statistics_.totalRecords
        << " selected_records=" << statistics_.selectedRecords << " file_bytes=" << statistics_.fileBytes
        << " source="<<(reused?"index.csv":"binary_scan")<<" elapsed_ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
HiltiBinReader::RawRecord HiltiBinReader::load(size_t record,bool readPayload) {
    const std::lock_guard<std::mutex> lock(inputMutex_);
    const auto& expected=records_.at(record); auto& input=inputs_.at(expected.fileId); seek(input,expected.offset);
    std::array<uint8_t,32> bytes; readExact(input,bytes.data(),bytes.size());
    RawRecord result; result.index=header(bytes,expected.offset);
    result.index.fileId=expected.fileId;
    if(result.index.kind!=expected.kind || result.index.sensorNs!=expected.sensorNs || result.index.bagNs!=expected.bagNs
        || (!expected.deferredHeader && (result.index.metadataBytes!=expected.metadataBytes || result.index.payloadBytes!=expected.payloadBytes))) fail("container changed after indexing");
    const auto& item=result.index; const auto fileBytes=fileSizes_.at(expected.fileId);
    if(expected.offset>fileBytes || fileBytes-expected.offset<32) fail("invalid indexed record offset");
    const uint64_t remaining=fileBytes-expected.offset-32;
    if(item.metadataBytes>limits_.maximumMetadataBytes || item.payloadBytes>limits_.maximumPayloadBytes
        || item.metadataBytes>remaining || item.payloadBytes>remaining-item.metadataBytes
        || (expected.deferredHeader && expected.recordBytes!=32+uint64_t(item.metadataBytes)+item.payloadBytes)) fail("invalid/changed indexed record size");
    result.index.camera=expected.camera;
    std::string json(item.metadataBytes,'\0'); readExact(input,json.data(),json.size()); result.metadata=YAML::Load(json);
    if(!result.metadata.IsMap()) fail("record metadata must be a JSON object");
    const auto topic=string(result.metadata,"topic"); const auto type=string(result.metadata,"msgtype");
    string(result.metadata,"frame_id"); integer(result.metadata,"connection_id");
    if(!fileTopics_[expected.fileId].empty() && topic!=fileTopics_[expected.fileId]) fail("loaded topic disagrees with indexed file");
    const char* expectedType=item.kind==RecordKind::Image?"sensor_msgs/msg/Image":item.kind==RecordKind::CompressedImage?
        "sensor_msgs/msg/CompressedImage":item.kind==RecordKind::Imu?"sensor_msgs/msg/Imu":"sensor_msgs/msg/PointCloud2";
    if(type!=expectedType || (item.kind==RecordKind::Imu && item.payloadBytes!=37*8)) fail("loaded sensor type/size mismatch");
    if(readPayload) { result.payload.resize(size_t(item.payloadBytes)); readExact(input,result.payload.data(),result.payload.size()); }
    return result;
}
ImageSample HiltiBinReader::readImage(size_t record) {
    auto raw=load(record); if(raw.index.camera<0) fail("record is not a selected image");
    ImageSample result; result.timestampNs=raw.index.sensorNs; result.bagNs=raw.index.bagNs;
    result.camera=size_t(raw.index.camera); result.topic=string(raw.metadata,"topic"); result.frameId=string(raw.metadata,"frame_id");
    if(raw.index.kind==RecordKind::CompressedImage) {
        result.sourceEncoding=string(raw.metadata,"format");
        result.image=cv::imdecode(raw.payload,cv::IMREAD_UNCHANGED);
    } else {
        const auto width=integer(raw.metadata,"width"),height=integer(raw.metadata,"height"),step=integer(raw.metadata,"step");
        const auto encoding=string(raw.metadata,"encoding"); result.sourceEncoding=encoding;
        boolean(raw.metadata,"is_bigendian"); // Every supported raw encoding uses single-byte channels.
        if(!width || !height || uint64_t(width)*height>limits_.maximumImagePixels
            || width>uint32_t(std::numeric_limits<int>::max()) || height>uint32_t(std::numeric_limits<int>::max())) fail("invalid image dimensions");
        int channels=1,code=-1;
        if(encoding=="mono8" || encoding=="8UC1") {}
        else if(encoding=="rgb8") { channels=3; code=cv::COLOR_RGB2BGR; }
        else if(encoding=="bgr8") channels=3;
        else if(encoding=="rgba8") { channels=4; code=cv::COLOR_RGBA2BGRA; }
        else if(encoding=="bgra8") channels=4;
        else if(encoding=="bayer_rggb8") code=cv::COLOR_BayerRGGB2BGR;
        else if(encoding=="bayer_bggr8") code=cv::COLOR_BayerBGGR2BGR;
        else if(encoding=="bayer_gbrg8") code=cv::COLOR_BayerGBRG2BGR;
        else if(encoding=="bayer_grbg8") code=cv::COLOR_BayerGRBG2BGR;
        else fail("unsupported image encoding: "+encoding+" (depth/16-bit images are not RGB camera input)");
        if(uint64_t(step)<uint64_t(width)*channels || uint64_t(height)*step!=raw.payload.size()) fail("image stride/payload mismatch");
        const cv::Mat view(int(height),int(width),CV_MAKETYPE(CV_8U,channels),raw.payload.data(),size_t(step));
        if(code>=0) cv::cvtColor(view,result.image,code); else result.image=view.clone();
    }
    if(result.image.empty() || result.image.depth()!=CV_8U || (result.image.channels()!=1 && result.image.channels()!=3 && result.image.channels()!=4)
        || result.image.total()>limits_.maximumImagePixels) fail("invalid decoded image");
    return result;
}
ImuSample HiltiBinReader::readImu(size_t record) {
    auto raw=load(record); if(raw.index.kind!=RecordKind::Imu || raw.payload.size()!=296) fail("invalid IMU payload");
    ImuSample result; result.timestampNs=raw.index.sensorNs; result.bagNs=raw.index.bagNs;
    result.topic=string(raw.metadata,"topic"); result.frameId=string(raw.metadata,"frame_id");
    for(size_t i=0;i<37;++i) result.original[i]=scalar<double>(raw.payload.data()+8*i,false);
    for(size_t i=0;i<3;++i) {
        result.gyro[i]=result.original[13+i]; result.acceleration[i]=result.original[25+i];
        if(!std::isfinite(result.gyro[i]) || !std::isfinite(result.acceleration[i])) fail("non-finite IMU gyro/acceleration");
    }
    return result;
}
LidarScan HiltiBinReader::readLidar(size_t record,bool decodePoints) {
    auto raw=load(record,decodePoints); if(raw.index.kind!=RecordKind::PointCloud2) fail("record is not a point cloud");
    LidarScan result; result.timestampNs=raw.index.sensorNs; result.bagNs=raw.index.bagNs;
    result.topic=string(raw.metadata,"topic"); result.frameId=string(raw.metadata,"frame_id");
    result.width=integer(raw.metadata,"width"); result.height=integer(raw.metadata,"height");
    result.pointStep=integer(raw.metadata,"point_step"); result.rowStep=integer(raw.metadata,"row_step");
    result.bigEndian=boolean(raw.metadata,"is_bigendian"); result.dense=boolean(raw.metadata,"is_dense");
    const uint64_t count=uint64_t(result.width)*result.height;
    if(!result.height || !result.pointStep || count>limits_.maximumScanPoints
        || uint64_t(result.rowStep)<uint64_t(result.width)*result.pointStep
        || uint64_t(result.rowStep)*result.height!=raw.index.payloadBytes) fail("point cloud dimensions/stride mismatch");
    const auto fields=raw.metadata["fields"];
    if(!fields.IsSequence() || fields.size()>256) fail("invalid PointCloud2 field table");
    std::map<std::string,size_t> byName;
    for(const auto& node:fields) {
        CloudField field{string(node,"name"),integer(node,"offset"),integer(node,"datatype"),integer(node,"count")};
        if(!field.count || uint64_t(field.offset)+uint64_t(field.count)*fieldBytes(field.datatype)>result.pointStep
            || !byName.emplace(field.name,result.fields.size()).second) fail("invalid/duplicate point field");
        result.fields.push_back(field);
    }
    const auto required=[&](const char* name)->const CloudField& {
        const auto it=byName.find(name); if(it==byName.end()) fail(std::string("missing Robot point field: ")+name);
        const auto& field=result.fields[it->second]; if(field.count!=1) fail("Robot scalar point field has count != 1"); return field;
    };
    const auto& x=required("x"); const auto& y=required("y"); const auto& z=required("z");
    const auto& intensity=required("intensity"); const auto& ring=required("ring"); const auto& timestamp=required("timestamp");
    if(timestamp.datatype!=8) fail("Robot absolute point timestamp must remain FLOAT64");
    // A VIO-only consumer needs scan provenance/counts, not unused point values.
    // Full consumers and --inspect retain payload reading and per-point validation.
    if(!decodePoints) return result;
    result.points.reserve(size_t(count));
    for(uint32_t row=0;row<result.height;++row) for(uint32_t column=0;column<result.width;++column) {
        const auto point=raw.payload.data()+uint64_t(row)*result.rowStep+uint64_t(column)*result.pointStep;
        const double px=pointScalar(point,x,result.bigEndian),py=pointScalar(point,y,result.bigEndian),pz=pointScalar(point,z,result.bigEndian);
        const double value=pointScalar(point,intensity,result.bigEndian),laser=pointScalar(point,ring,result.bigEndian);
        const double seconds=pointScalar(point,timestamp,result.bigEndian);
        if(!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) || !std::isfinite(value) || !std::isfinite(seconds)) { ++result.invalidPoints; continue; }
        if(!std::isfinite(laser) || laser<0 || laser>65535 || laser!=std::floor(laser)) fail("invalid LiDAR ring");
        LidarPoint decoded; decoded.x=float(px); decoded.y=float(py); decoded.z=float(pz); decoded.intensity=float(value);
        if(!std::isfinite(decoded.x) || !std::isfinite(decoded.y) || !std::isfinite(decoded.z) || !std::isfinite(decoded.intensity)) fail("point cannot be represented in float32");
        decoded.ring=uint16_t(laser); decoded.timestampNs=pointTime(seconds);
        // Robot scan header is its last point. Reject a relative-time convention
        // or clock mismatch instead of silently inventing per-point timing.
        const long double relative=static_cast<long double>(decoded.timestampNs)-result.timestampNs;
        if(relative < -1000000000.0L || relative > 2000000.0L) fail("point time is inconsistent with Robot last-point scan header");
        if(result.points.empty()) result.firstPointNs=result.lastPointNs=decoded.timestampNs;
        else { result.firstPointNs=std::min(result.firstPointNs,decoded.timestampNs); result.lastPointNs=std::max(result.lastPointNs,decoded.timestampNs); }
        result.points.push_back(decoded);
    }
    result.originalData=std::move(raw.payload); return result;
}

std::vector<ImagePacketIndex> makeImagePackets(const HiltiBinReader& reader,int64_t toleranceNs,
    ImageGroupingStatistics* statistics) {
    if(toleranceNs<0 || toleranceNs>50000000) fail("image bundle tolerance must be between 0 and 50 ms");
    std::array<size_t,4> cursors{}; std::vector<ImagePacketIndex> packets; ImageGroupingStatistics result;
    const auto& cameras=reader.cameras(); const auto& records=reader.records();
    size_t totalImages=0;
    for(const auto& camera:cameras) {
        totalImages+=camera.size();
        for(size_t i=1;i<camera.size();++i) {
            // A window smaller than half a stream's frame spacing prevents a
            // bundle from containing two plausible candidates from that camera.
            if(2*toleranceNs>=records[camera[i]].sensorNs-records[camera[i-1]].sensorNs)
                fail("image bundle tolerance must be smaller than half every camera frame interval");
        }
    }
    for(;;) {
        bool exhausted=false;
        for(size_t c=0;c<4;++c) exhausted=exhausted || cursors[c]==cameras[c].size();
        if(exhausted) break;
        ImagePacketIndex packet;
        for(size_t c=0;c<4;++c) packet.imageTimestampNs[c]=records[cameras[c][cursors[c]]].sensorNs;
        packet.firstImageNs=*std::min_element(packet.imageTimestampNs.begin(),packet.imageTimestampNs.end());
        packet.lastImageNs=*std::max_element(packet.imageTimestampNs.begin(),packet.imageTimestampNs.end());
        if(packet.lastImageNs-packet.firstImageNs>toleranceNs) {
            // The oldest images cannot match the newer head of every stream.
            // Advance them to recover after a missing camera frame, without reusing
            // an image or silently emitting a partial camera packet.
            for(size_t c=0;c<4;++c) if(packet.imageTimestampNs[c]==packet.firstImageNs) ++cursors[c];
            continue;
        }
        int64_t offsetSum=0;
        for(size_t c=0;c<4;++c) {
            offsetSum+=packet.imageTimestampNs[c]-packet.firstImageNs;
            packet.records.push_back(cameras[c][cursors[c]++]);
        }
        // Subtract the epoch first: summing absolute ns would overflow int64 and
        // converting to floating-point seconds would lose sub-microsecond precision.
        packet.timestampNs=packet.firstImageNs+(offsetSum+2)/4;
        result.maxBundleSpanNs=std::max(result.maxBundleSpanNs,packet.lastImageNs-packet.firstImageNs);
        for(const auto time:packet.imageTimestampNs)
            result.maxCameraOffsetNs=std::max(result.maxCameraOffsetNs,std::abs(time-packet.timestampNs));
        packets.push_back(std::move(packet));
    }
    result.unmatchedImages=totalImages-4*packets.size();
    if(statistics) *statistics=result;
    return packets;
}
} }

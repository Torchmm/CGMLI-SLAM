// Standalone reader regression; intentionally excluded from the application.
// Link reader sources, project camera/time support, yaml-cpp, Eigen and OpenCV.
#include <offline/HiltiBinReader.hpp>
#include <offline/HiltiDatasetReader.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/NoDistortion.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>

namespace {
using namespace cgmli::offline;
using Bytes=std::vector<uint8_t>;
void require(bool condition,const char* reason) { if(!condition) throw std::runtime_error(reason); }
template<class T> void put(Bytes& bytes,size_t offset,T value,bool big=false) {
    uint8_t raw[sizeof(T)]; std::memcpy(raw,&value,sizeof(T)); const uint16_t one=1;
    const bool hostBig=*reinterpret_cast<const uint8_t*>(&one)==0;
    if(hostBig!=big) std::reverse(raw,raw+sizeof(T));
    if(bytes.size()<offset+sizeof(T)) bytes.resize(offset+sizeof(T));
    std::copy(raw,raw+sizeof(T),bytes.begin()+offset);
}
template<class T> void append(Bytes& bytes,T value) { put(bytes,bytes.size(),value); }
void record(Bytes& file,uint32_t kind,int64_t time,const std::string& json,const Bytes& payload) {
    append(file,kind); append(file,uint32_t(json.size())); append(file,time); append(file,time+1000); append(file,uint64_t(payload.size()));
    file.insert(file.end(),json.begin(),json.end()); file.insert(file.end(),payload.begin(),payload.end());
}
std::string metadata(const std::string& topic,const std::string& type,const std::string& extra) {
    return "{\"topic\":\""+topic+"\",\"msgtype\":\"sensor_msgs/msg/"+type
        +"\",\"frame_id\":\"fixture\",\"connection_id\":1"+extra+"}";
}
constexpr int64_t epoch=1670000000000000000LL;
Bytes fixture(bool duplicate=false,bool bigCloud=false,const std::vector<std::array<int64_t,4>>& bundles=
    {{epoch,epoch+1000000,epoch+2000000,epoch+3000000}}) {
    Bytes file{'H','I','L','T','B','I','N','1'}; append(file,uint32_t(1)); HiltiTopics topics;
    // Sensor order differs from write/bag order; RGB rows include two pad bytes.
    for(const auto& times:bundles) for(int c=3;c>=0;--c) {
        if(times[size_t(c)]<0) continue;
        const auto json=metadata(topics.cameras[size_t(c)],"Image",",\"width\":2,\"height\":1,\"step\":8,\"encoding\":\"rgb8\",\"is_bigendian\":0");
        record(file,1,times[size_t(c)],json,{10,20,30,40,50,60,123,124});
        if(duplicate && c==0) record(file,1,epoch,json,{10,20,30,40,50,60,123,124});
    }
    Bytes imu; for(int i=0;i<37;++i) append(imu,double(i)+0.25);
    record(file,4,epoch+100000000,metadata(topics.imu,"Imu",""),imu);
    record(file,4,epoch-500000000,metadata(topics.imu,"Imu",""),imu);
    Bytes cloud(80,0xcc);
    for(size_t row=0;row<2;++row) {
        const size_t base=row*40;
        put(cloud,base,float(1+row),bigCloud); put(cloud,base+4,float(2),bigCloud); put(cloud,base+8,float(3),bigCloud);
        put(cloud,base+12,float(42),bigCloud); put(cloud,base+16,uint16_t(127+row),bigCloud);
        put(cloud,base+24,double(epoch)*1e-9-double(1-row)*0.01,bigCloud);
    }
    const auto fields=",\"width\":1,\"height\":2,\"point_step\":32,\"row_step\":40,\"is_dense\":true,\"is_bigendian\":"
        +std::string(bigCloud?"true":"false")+",\"fields\":["
        "{\"name\":\"x\",\"offset\":0,\"datatype\":7,\"count\":1},"
        "{\"name\":\"y\",\"offset\":4,\"datatype\":7,\"count\":1},"
        "{\"name\":\"z\",\"offset\":8,\"datatype\":7,\"count\":1},"
        "{\"name\":\"intensity\",\"offset\":12,\"datatype\":7,\"count\":1},"
        "{\"name\":\"ring\",\"offset\":16,\"datatype\":4,\"count\":1},"
        "{\"name\":\"timestamp\",\"offset\":24,\"datatype\":8,\"count\":1}]";
    record(file,3,epoch,metadata(topics.lidar,"PointCloud2",fields),cloud); return file;
}
void write(const std::filesystem::path& path,const Bytes& data) {
    std::ofstream file(path,std::ios::binary); file.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    require(bool(file),"Fixture write failed");
}
uint64_t little(const Bytes& bytes,size_t offset,size_t count) {
    uint64_t value=0; for(size_t i=0;i<count;++i) value|=uint64_t(bytes.at(offset+i))<<(8*i); return value;
}
void splitFixture(const std::filesystem::path& directory,const Bytes& combined) {
    std::filesystem::create_directories(directory); std::map<std::string,Bytes> streams;
    for(size_t offset=12;offset<combined.size();) {
        const size_t metadataBytes=size_t(little(combined,offset+4,4));
        const size_t bytes=32+metadataBytes+size_t(little(combined,offset+24,8));
        const std::string json(combined.begin()+offset+32,combined.begin()+offset+32+metadataBytes);
        auto& output=streams[YAML::Load(json)["topic"].as<std::string>()];
        if(output.empty()) output.assign(combined.begin(),combined.begin()+12);
        output.insert(output.end(),combined.begin()+offset,combined.begin()+offset+bytes); offset+=bytes;
    }
    std::ofstream receipt(directory/"summary.json");
    receipt<<"{\"complete\":true,\"format\":\"HILTBIN1\",\"version\":1,\"topic_files\":{";
    size_t file=0;
    for(const auto& stream:streams) {
        const auto filename=std::to_string(file)+".bin"; write(directory/filename,stream.second);
        if(file++) receipt<<','; receipt<<'\"'<<stream.first<<"\":\""<<filename<<'\"';
    }
    receipt<<"}}"; require(bool(receipt),"Directory receipt write failed");
}
void rejects(const std::function<void()>& operation) {
    bool threw=false; try { operation(); } catch(const std::exception&) { threw=true; }
    require(threw,"Malformed input was accepted");
}
void converterCsvFastPath(const std::filesystem::path& directory) {
    splitFixture(directory,fixture());
    std::ifstream manifest(directory/"summary.json"); const auto summary=YAML::Load(manifest);
    const auto csvPath=directory/"index.csv";
    {
        std::ofstream csv(csvPath); csv<<"bag_time_ns,sensor_time_ns,kind,connection_id,topic,file,offset,record_bytes\n";
        for(const auto& entry:summary["topic_files"]) {
            const auto topic=entry.first.as<std::string>(),name=entry.second.as<std::string>();
            std::ifstream file(directory/name,std::ios::binary);
            Bytes bytes(size_t(std::filesystem::file_size(directory/name))); file.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()));
            for(size_t offset=12;offset<bytes.size();) {
                const size_t length=32+size_t(little(bytes,offset+4,4))+size_t(little(bytes,offset+24,8));
                csv<<little(bytes,offset+16,8)<<','<<little(bytes,offset+8,8)<<','<<little(bytes,offset,4)
                    <<",1,"<<topic<<','<<name<<','<<offset<<','<<length<<'\n'; offset+=length;
            }
        }
    }
    std::filesystem::last_write_time(csvPath,std::filesystem::file_time_type::clock::now()+std::chrono::seconds(1));
    {
        HiltiBinReader reader(directory);
        require(reader.statistics().reusedCsv && reader.records().size()==7,"completed converter CSV should avoid binary scan");
        require(reader.readImu(reader.imu()[0]).gyro[0]==13.25,"deferred IMU header/payload validation failed");
        require(reader.readImage(reader.cameras()[0][0]).image.at<cv::Vec3b>(0,0)==cv::Vec3b(30,20,10),"CSV image offset is wrong");
        const auto metadata=reader.readLidar(reader.lidar()[0],false);
        require(metadata.width==1 && metadata.height==2 && metadata.points.empty() && metadata.originalData.empty(),
            "metadata-only cloud read must retain layout without payload allocation");
        require(reader.readLidar(reader.lidar()[0]).points.size()==2,"full cloud API must retain default point decoding");
    }
    {
        std::ofstream csv(csvPath,std::ios::trunc); csv<<"invalid schema\n";
    }
    HiltiBinReader fallback(directory);
    require(!fallback.statistics().reusedCsv && fallback.records().size()==7,"malformed CSV must fall back to checked binary scan");
}
cgmli::ViParameters readerParameters() {
    cgmli::ViParameters result{}; result.imu.use=true; result.imu.max_sample_gap=1.0;
    for(size_t c=0;c<4;++c) result.nCameraSystem.addCamera(std::make_shared<cgmli::kinematics::Transformation>(),
        std::make_shared<cgmli::cameras::PinholeCamera<cgmli::cameras::NoDistortion>>(2,1,1,1,1,0.5,cgmli::cameras::NoDistortion()),
        cgmli::cameras::NCameraSystem::NoDistortion,false);
    return result;
}
void waitForReader(HiltiDatasetReader& reader) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(reader.isStreaming() && std::chrono::steady_clock::now()<end) std::this_thread::yield();
    const bool finished=!reader.isStreaming(); reader.stopStreaming(); require(finished,"Reader did not finish");
}
void streamingCallbacks(const std::filesystem::path& directory) {
    auto parameters=readerParameters(); HiltiDatasetReader reader(directory,parameters);
    size_t images=0,imu=0,scans=0; std::vector<uint64_t> times;
    reader.setImagesCallback([&](const cgmli::Time& time,const std::map<size_t,cv::Mat>& frames,const std::map<size_t,cgmli::Time>& originalTimes) {
        require(frames.size()==4 && originalTimes.size()==4,"Reader must deliver complete four-camera bundles");
        for(size_t c=0;c<4;++c) require(originalTimes.at(c).toNSec()==uint64_t(epoch+int64_t(c)*1000000),"Original camera timestamp lost");
        images+=frames.size(); times.push_back(time.toNSec()); return true;
    });
    reader.setImuCallback([&](const cgmli::Time&,const Eigen::Vector3d& acceleration,const Eigen::Vector3d& gyro) {
        require(acceleration.x()==25.25 && gyro.x()==13.25,"Callback IMU units/layout changed"); ++imu; return true;
    });
    reader.setLidarCallback([&](const LidarScan& scan){ require(scan.points.size()==2,"Callback lost point cloud"); ++scans; return true; });
    reader.startStreaming(); waitForReader(reader); reader.rethrowIfFailed();
    require(images==4 && imu==2 && scans==1,"Callbacks did not deliver all six streams");
    require(times.size()==1 && times.front()==uint64_t(epoch+1500000),"Node time must be the four-camera midpoint mean");
    require(reader.statistics().grouping.maxBundleSpanNs==3000000
        && reader.statistics().grouping.maxCameraOffsetNs==1500000,"Bundle timing diagnostics failed");
    require(reader.completion()==1.0 && reader.statistics().imageObservations==4,"Reader EOF statistics failed");

    parameters.imu.max_sample_gap=0.05; HiltiDatasetReader gap(directory,parameters);
    gap.setImagesCallback([](const cgmli::Time&,const std::map<size_t,cv::Mat>&,const std::map<size_t,cgmli::Time>&){return true;});
    gap.setImuCallback([](const cgmli::Time&,const Eigen::Vector3d&,const Eigen::Vector3d&){return true;});
    gap.setLidarCallback([](const LidarScan&){return true;});
    gap.startStreaming(); waitForReader(gap); rejects([&]{gap.rethrowIfFailed();});
}
void stopBlockedCallback(const std::filesystem::path& directory) {
    HiltiDatasetReader reader(directory,readerParameters());
    std::mutex mutex; std::condition_variable changed; bool entered=false,released=false;
    reader.setImagesCallback([&](const cgmli::Time&,const std::map<size_t,cv::Mat>&,const std::map<size_t,cgmli::Time>&) {
        std::unique_lock<std::mutex> lock(mutex); entered=true; changed.notify_all();
        changed.wait(lock,[&]{return released;}); return false;
    });
    reader.setImuCallback([](const cgmli::Time&,const Eigen::Vector3d&,const Eigen::Vector3d&){return true;});
    reader.setLidarCallback([](const LidarScan&){return true;});
    reader.startStreaming(); bool reached=false;
    { std::unique_lock<std::mutex> lock(mutex); reached=changed.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}); }
    reader.requestStop();
    { std::lock_guard<std::mutex> lock(mutex); released=true; }
    changed.notify_all(); reader.stopStreaming(); reader.rethrowIfFailed();
    require(reached && !reader.isStreaming(),"Cancellation failed to release the blocking callback");
}
void meanTimeBundles(const std::filesystem::path& path) {
    write(path,fixture(false,false,{{epoch+1,epoch+10002001,epoch+1,epoch+10002001}}));
    {
        HiltiBinReader reader(path); ImageGroupingStatistics stats;
        const auto packets=makeImagePackets(reader,25000000,&stats);
        require(packets.size()==1 && packets.front().timestampNs==epoch+5001001,"Mean must preserve integer ns at Unix epoch scale");
        require(stats.maxCameraOffsetNs==5001000 && stats.unmatchedImages==0,"Symmetric midpoint offsets incorrect");
        const auto strict=makeImagePackets(reader,0);
        require(strict.empty(),"Zero tolerance must reject non-simultaneous midpoints rather than emit partial bundles");
    }
    write(path,fixture(false,false,{{epoch,epoch,epoch,epoch+2}}));
    { HiltiBinReader reader(path); require(makeImagePackets(reader).front().timestampNs==epoch+1,"Half-ns mean rounding failed"); }
    write(path,fixture(false,false,{
        {epoch,epoch+10002000,epoch,epoch+10002000},
        {epoch+100000000,-1,epoch+100000000,epoch+110002000},
        {epoch+200000000,epoch+210002000,epoch+200000000,epoch+210002000}}));
    {
        HiltiBinReader reader(path); ImageGroupingStatistics stats;
        const auto packets=makeImagePackets(reader,25000000,&stats);
        require(packets.size()==2 && stats.unmatchedImages==3,"Missing camera frame must discard only unmatched images and recover");
        require(packets[0].timestampNs==epoch+5001000 && packets[1].timestampNs==epoch+205001000,"Cross-trigger pairing or mean failed");
        std::set<size_t> used;
        for(const auto& packet:packets) {
            require(packet.records.size()==4,"Incomplete bundle emitted");
            for(auto record:packet.records) require(used.insert(record).second,"Image reused across bundles");
        }
        rejects([&]{makeImagePackets(reader,50000000);});
    }
    write(path,fixture(false,false,{{epoch,epoch,epoch,epoch+40000000}}));
    { HiltiBinReader reader(path); ImageGroupingStatistics stats;
        require(makeImagePackets(reader,25000000,&stats).empty() && stats.unmatchedImages==4,"Excessive span must not form a bundle"); }
}
}
int main() {
    const auto root=std::filesystem::temp_directory_path()/std::filesystem::u8path(u8"cgmli_离线_")
        /std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    try {
        std::filesystem::create_directories(root); const auto path=root/"data.bin";
        converterCsvFastPath(root/"csv_fast_path");
        write(path,fixture());
        {
            HiltiBinReader reader(path);
            require(reader.records().size()==7,"Wrong selected record count");
            const auto image=reader.readImage(reader.cameras()[0][0]);
            require(image.image.type()==CV_8UC3 && image.image.cols==2,"Wrong image layout");
            require(image.image.at<cv::Vec3b>(0,0)==cv::Vec3b(30,20,10),"RGB conversion or row stride failed");
            const auto imu=reader.readImu(reader.imu()[0]);
            require(imu.timestampNs==epoch-500000000 && imu.gyro[0]==13.25 && imu.acceleration[0]==25.25,"IMU indexing/order failed");
            require(imu.original[36]==36.25,"Covariance preservation failed");
            const auto scan=reader.readLidar(reader.lidar()[0]);
            require(scan.points.size()==2 && scan.points[1].x==2 && scan.points[1].ring==128,"Cloud stride/field decode failed");
            require(scan.originalData.size()==80 && scan.originalData[39]==0xcc,"Cloud padding lost");
            require(std::abs(scan.points.back().timestampNs-epoch)<=300,"Absolute double timestamp lost precision");
            require(makeImagePackets(reader).size()==1,"Default must form one complete four-camera bundle");
            const auto packets=makeImagePackets(reader,3000000);
            require(packets.size()==1 && packets[0].records.size()==4 && packets[0].lastImageNs-epoch==3000000
                && packets[0].timestampNs==epoch+1500000,"Explicit mean-time grouping failed");
            rejects([&]{makeImagePackets(reader,-1);});
        }
        meanTimeBundles(path);
        write(path,fixture(false,true));
        { HiltiBinReader reader(path); require(reader.readLidar(reader.lidar()[0]).points[1].ring==128,"Big-endian cloud failed"); }
        const auto directory=root/"per_topic"; splitFixture(directory,fixture(false,true));
        streamingCallbacks(directory); stopBlockedCallback(directory);
        {
            HiltiBinReader reader(directory);
            require(reader.records().size()==7 && reader.readImu(reader.imu()[0]).timestampNs==epoch-500000000,"Cross-file indexing/order failed");
            for(size_t c=0;c<4;++c) require(reader.readImage(reader.cameras()[c][0]).timestampNs==epoch+int64_t(c)*1000000,"File offset/camera mapping failed");
            require(reader.readLidar(reader.lidar()[0]).points.back().ring==128,"Per-topic cloud decode failed");
        }
        {
            std::ofstream receipt(directory/"summary.json");
            receipt<<"{\"complete\":true,\"format\":\"HILTBIN1\",\"version\":1,\"topic_files\":{\"/imu/data\":\"../data.bin\"}}";
        }
        rejects([&]{HiltiBinReader reader(directory);});
        write(path,fixture(true)); rejects([&]{HiltiBinReader reader(path);});
        auto truncated=fixture(); truncated.pop_back(); write(path,truncated); rejects([&]{HiltiBinReader reader(path);});
        write(path,fixture());
        ReadLimits limits; limits.maximumMetadataBytes=10; rejects([&]{HiltiBinReader reader(path,{},limits);});
        const auto partial=root/"data.bin.partial"; write(partial,fixture()); rejects([&]{HiltiBinReader reader(partial);});
        { std::ofstream receipt(root/"summary.json"); receipt<<"{\"complete\":false}"; }
        rejects([&]{HiltiBinReader reader(path);});
        std::filesystem::remove_all(root); std::cout<<"Hilti reader regression passed\n"; return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; std::filesystem::remove_all(root); return 1;
    }
}

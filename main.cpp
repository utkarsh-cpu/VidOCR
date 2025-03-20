#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <map>
#include <utility>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif


using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct ChunkData {
    std::string videoPath;
    std::string audioPath;
    int chunkNum;
    int videoIndex;
    std::string baseName;
};

string executeCommand(const std::string& command) {
    string result;
#ifdef _WIN32
    HANDLE pipe;
        SECURITY_ATTRIBUTES saAttr;
        saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
        saAttr.bInheritHandle = TRUE;
        saAttr.lpSecurityDescriptor = NULL;

        if (!CreatePipe(&pipe, &saAttr, &saAttr, 0)) {
            throw std::runtime_error("CreatePipe failed");
        }

        STARTUPINFOA si = { sizeof(STARTUPINFOA) };
        PROCESS_INFORMATION pi;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = pipe;
        si.hStdError = pipe;

        char* cmd_array = const_cast<char*>(command.c_str()); // Need char* for CreateProcessA
        if (!CreateProcessA(NULL, cmd_array, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
            CloseHandle(pipe);
            throw std::runtime_error("CreateProcessA failed");
        }

        CloseHandle(pipe); // Close the write end of the pipe in the parent

        char buffer[128];
        DWORD bytesRead;
        HANDLE readPipe = GetStdHandle(STD_OUTPUT_HANDLE); // Get handle to the read end (incorrect, should use pipe handle)
        if (!DuplicateHandle(GetCurrentProcess(), pipe, GetCurrentProcess(), &readPipe, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            throw std::runtime_error("DuplicateHandle failed");
        }

        while (ReadFile(readPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
            buffer[bytesRead] = '\0';
            result += buffer;
        }

        CloseHandle(readPipe);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

#else
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen failed!");
    }
    char buffer[128];
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr) {
            result += buffer;
        }
    }
    pclose(pipe);
#endif
    return result;
}


class Video {
private:
    string video_path;

    static double calculate_laplacian_variance(Mat &img) {
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Laplacian(gray, gray, CV_8U, 1);
        Scalar mean, dev;
        meanStdDev(gray, mean, dev);
        return dev.val[0] * dev.val[0];
    }

    void denoising_video() {
        cout << "[DEBUG] Starting video denoising function" << endl;

        // Open video
        cout << "[DEBUG] Attempting to open video file: " << video_path << endl;
        VideoCapture video(video_path);
        if (!video.isOpened()) {
            cout << "Error opening video file" << endl;
            return;
        }

        // Setup output file
        string file_name = video_path;
        size_t last_dot = file_name.find_last_of('.');
        std::string outputFile = file_name.substr(0, last_dot) + "_processed.mp4";
        cout << "[DEBUG] Output file will be: " << outputFile << endl;

        int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = video.get(cv::CAP_PROP_FPS);
        cout << "[DEBUG] Video properties - Width: " << width << ", Height: " << height << ", FPS: " << fps << endl;

        VideoWriter writer(outputFile,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps,
                           cv::Size(width, height));

        // Create debug directory
        std::string debug_dir = file_name.substr(0, last_dot) + "_debug";
        std::filesystem::create_directory(debug_dir);
        cout << "[DEBUG] Created debug directory: " << debug_dir << endl;

        // Thread management
        const unsigned int num_workers = std::thread::hardware_concurrency();
        cout << "[DEBUG] Using " << num_workers << " worker threads" << endl;

        std::vector<std::thread> workers;
        std::queue<std::pair<size_t, cv::Mat>> processed_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::atomic<size_t> frame_index{0};
        std::atomic<bool> done{false};
        std::queue<std::pair<size_t, cv::Mat>> raw_frames;

        // Statistics tracking
        std::atomic<size_t> total_frames_read{0};
        std::atomic<size_t> total_frames_processed{0};
        std::atomic<size_t> total_frames_written{0};
        std::atomic<size_t> total_sharpened_frames{0};

        // Worker threads
        cout << "[DEBUG] Starting worker threads..." << endl;
        workers.reserve(num_workers);
        for(unsigned int i = 0; i < num_workers; ++i) {
            workers.emplace_back([&, worker_id=i] {
                cout << "[DEBUG] Worker " << worker_id << " started" << endl;
                size_t frames_processed = 0;

                while(true) {
                    // Get a frame to process
                    cv::Mat frame;
                    size_t current_index;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cout << "[DEBUG] Worker " << worker_id << " waiting for frame" << endl;
                        queue_cv.wait(lock, [&] {
                            return !raw_frames.empty() || done;
                        });

                        if(raw_frames.empty() && done) {
                            cout << "[DEBUG] Worker " << worker_id << " exiting, no more frames" << endl;
                            break;
                        }

                        if(!raw_frames.empty()) {
                            auto [idx, f] = raw_frames.front();
                            raw_frames.pop();
                            frame = f;
                            current_index = idx;
                            cout << "[DEBUG] Worker " << worker_id << " processing frame " << current_index << endl;
                        } else {
                            continue;
                        }
                    }

                    // Process the frame
                    double variance = calculate_laplacian_variance(frame);
                    cout << "[DEBUG] Worker " << worker_id << " - Frame " << current_index
                         << " Laplacian variance: " << variance << endl;

                    bool was_sharpened = false;
                    if(variance < 100) {
                        cout << "[DEBUG] Worker " << worker_id << " - Frame " << current_index
                             << " requires sharpening" << endl;

                        // Save before image for comparison
                        cv::Mat before_frame = frame.clone();
                        std::string before_path = debug_dir + "/frame_" + std::to_string(current_index) + "_before.jpg";
                        cv::imwrite(before_path, before_frame);

                        // Apply sharpening
                        Mat sharpen_kernel = (Mat_<double>(3,3) <<
                                                                -1, -1, -1,
                                -1, 9, -1,
                                -1, -1, -1);
                        filter2D(frame, frame, -1, sharpen_kernel);
                        was_sharpened = true;
                        total_sharpened_frames++;

                        // Save after image
                        std::string after_path = debug_dir + "/frame_" + std::to_string(current_index) + "_after.jpg";
                        cv::imwrite(after_path, frame);
                    }

                    // Add processed frame to queue
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        processed_queue.emplace(current_index, frame);
                        cout << "[DEBUG] Worker " << worker_id << " added frame " << current_index
                             << " to processed queue" << (was_sharpened ? " (sharpened)" : "") << endl;
                    }

                    frames_processed++;
                    total_frames_processed++;
                    cout << "[DEBUG] Worker " << worker_id << " has processed " << frames_processed
                         << " frames total" << endl;

                    queue_cv.notify_one();
                }
            });
        }

        // Writer thread
        cout << "[DEBUG] Starting writer thread" << endl;
        std::thread writer_thread([&] {
            size_t expected_index = 0;
            std::map<size_t, cv::Mat> ordered_buffer;
            size_t frames_written = 0;

            while(true) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cout << "[DEBUG] Writer waiting for processed frames, expected index: " << expected_index << endl;
                queue_cv.wait(lock, [&] {
                    return !processed_queue.empty() || (done && ordered_buffer.empty() && processed_queue.empty());
                });

                if(processed_queue.empty() && ordered_buffer.empty() && done) {
                    cout << "[DEBUG] Writer thread exiting, all frames written" << endl;
                    break;
                }

                cout << "[DEBUG] Writer processing " << processed_queue.size() << " frames in queue, "
                     << ordered_buffer.size() << " frames in buffer" << endl;

                while(!processed_queue.empty()) {
                    auto [idx, frame] = processed_queue.front();
                    processed_queue.pop();
                    cout << "[DEBUG] Writer got frame " << idx << " (expected: " << expected_index << ")" << endl;

                    if(idx == expected_index) {
                        writer.write(frame);
                        frames_written++;
                        total_frames_written++;
                        cout << "[DEBUG] Writer wrote frame " << idx << " directly" << endl;
                        expected_index++;

                        // Check buffer for subsequent frames
                        while(ordered_buffer.count(expected_index)) {
                            writer.write(ordered_buffer[expected_index]);
                            frames_written++;
                            total_frames_written++;
                            cout << "[DEBUG] Writer wrote buffered frame " << expected_index << endl;
                            ordered_buffer.erase(expected_index);
                            expected_index++;
                        }

                        cout << "[DEBUG] Writer has written " << frames_written << " frames total" << endl;
                    } else {
                        ordered_buffer[idx] = frame;
                        cout << "[DEBUG] Writer buffered out-of-order frame " << idx << endl;
                    }
                }
            }
        });

        // Main thread as producer
        cout << "[DEBUG] Starting main thread as producer" << endl;
        size_t idx = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while(video.isOpened()) {
            cv::Mat frame;
            if(!video.read(frame)) {
                cout << "[DEBUG] End of video reached after " << idx << " frames" << endl;
                break;
            }

            total_frames_read++;

            if(idx % 100 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                cout << "[DEBUG] Progress: " << idx << " frames read ("
                     << (elapsed > 0 ? idx / elapsed : 0) << " fps)" << endl;

                // Create status report file
                std::ofstream status_report(debug_dir + "/status_report.txt", std::ios::app);
                status_report << "Frame: " << idx
                              << ", Read: " << total_frames_read
                              << ", Processed: " << total_frames_processed
                              << ", Sharpened: " << total_sharpened_frames
                              << ", Written: " << total_frames_written
                              << ", Elapsed: " << elapsed << "s" << std::endl;
                status_report.close();
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                raw_frames.emplace(idx, frame.clone());
                cout << "[DEBUG] Producer added frame " << idx << " to raw queue" << endl;
            }
            queue_cv.notify_one();
            idx++;
        }

        // Signal completion
        cout << "[DEBUG] All frames read, signaling completion" << endl;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            done = true;
        }
        queue_cv.notify_all();

        // Cleanup
        cout << "[DEBUG] Waiting for worker threads to finish" << endl;
        for(auto& t : workers) {
            if(t.joinable()) t.join();
        }
        cout << "[DEBUG] All worker threads joined" << endl;

        cout << "[DEBUG] Waiting for writer thread to finish" << endl;
        if(writer_thread.joinable()) writer_thread.join();
        cout << "[DEBUG] Writer thread joined" << endl;

        video.release();
        writer.release();

        // Final statistics
        cout << "[DEBUG] Final statistics:" << endl;
        cout << "[DEBUG] Total frames read: " << total_frames_read << endl;
        cout << "[DEBUG] Total frames processed: " << total_frames_processed << endl;
        cout << "[DEBUG] Total frames sharpened: " << total_sharpened_frames
             << " (" << (total_frames_processed > 0 ? (total_sharpened_frames * 100.0 / total_frames_processed) : 0)
             << "%)" << endl;
        cout << "[DEBUG] Total frames written: " << total_frames_written << endl;

        // Create final report
        std::ofstream final_report(debug_dir + "/final_report.txt");
        final_report << "Video Processing Report" << std::endl;
        final_report << "======================" << std::endl;
        final_report << "Input: " << video_path << std::endl;
        final_report << "Output: " << outputFile << std::endl;
        final_report << "Resolution: " << width << "x" << height << std::endl;
        final_report << "FPS: " << fps << std::endl;
        final_report << "Worker threads: " << num_workers << std::endl;
        final_report << "Total frames processed: " << total_frames_processed << std::endl;
        final_report << "Total frames sharpened: " << total_sharpened_frames
                     << " (" << (total_frames_processed > 0 ? (total_sharpened_frames * 100.0 / total_frames_processed) : 0)
                     << "%)" << std::endl;
        final_report.close();

        cout << "Video processing completed. Debug information available in: " << debug_dir << endl;
    }

    static bool isFFmpegAvailable() {
        try {
            executeCommand("ffmpeg -version");
            return true;
        } catch (const std::runtime_error& e) {
            return false;
        }
    }

    static vector<ChunkData> chunkVideo(const string& videoPath, int chunkDuration, int videoIndex, const string& baseName) {
        if (!isFFmpegAvailable()) {
            throw std::runtime_error("ffmpeg not found in PATH");
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        uniform_int_distribution<> distrib(0, INT_MAX);
        fs::path tempDir = fs::temp_directory_path() / fs::path("video_chunks_" + to_string(distrib(gen))); // Create unique temp dir name
        fs::create_directories(tempDir);
        if (!fs::exists(tempDir)) {
            throw std::runtime_error("Error creating temporary directory");
        }

        std::string ffprobeCommand = "ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"" + videoPath + "\"";
        std::string durationOutput;
        try {
            durationOutput = executeCommand(ffprobeCommand);
        } catch (const std::runtime_error& e) {
            fs::remove_all(tempDir);
            throw std::runtime_error("Error getting video duration: " + std::string(e.what()) + ", output: " + durationOutput);
        }

        double duration;
        try {
            duration = std::stod(durationOutput);
        } catch (const std::invalid_argument& e) {
            fs::remove_all(tempDir);
            throw std::runtime_error("Error parsing video duration: " + std::string(e.what()));
        }

        int numChunks = static_cast<int>(duration / static_cast<double>(chunkDuration));
        if (static_cast<int>(duration) % chunkDuration != 0) {
            numChunks++;
        }

        std::vector<ChunkData> chunks;
        for (int i = 0; i < numChunks; ++i) {
            int startTime = i * chunkDuration;
            std::string chunkVideoPath = (tempDir / ("chunk_" + std::to_string(i) + "_video_" + std::to_string(videoIndex) + ".mp4")).string();
            std::string chunkAudioPath = (tempDir / ("chunk_" + std::to_string(i) + "_video_" + std::to_string(videoIndex) + ".wav")).string();

            std::stringstream ffmpegCmdStream;
            ffmpegCmdStream << "ffmpeg "
                            << "-ss " << to_string(startTime) << " "
                            << "-i \"" << videoPath << "\" "
                            << "-t " << to_string(chunkDuration) << " "
                            << "-c copy -an \"" << chunkVideoPath << "\" "
                            << "-ss " << to_string(startTime) << " "
                            << "-i \"" << videoPath << "\" "
                            << "-t " << to_string(chunkDuration) << " "
                            << "-vn -acodec pcm_s16le \"" << chunkAudioPath << "\"";
            std::string ffmpegCommand = ffmpegCmdStream.str();

            std::string chunkOutput;
            try {
                chunkOutput = executeCommand(ffmpegCommand);
            } catch (const std::runtime_error& e) {
                // Clean up temp dir on chunking error, though ideally we'd only remove the failed chunk
                fs::remove_all(tempDir);
                throw std::runtime_error("Error creating video chunk " + std::to_string(i) + " for video " + std::to_string(videoIndex) + ": " + std::string(e.what()) + ", output: " + chunkOutput);
            }

            chunks.push_back({chunkVideoPath, chunkAudioPath, i, videoIndex, baseName});
        }

        return chunks;
    }
public:
    explicit Video(string  path) : video_path(std::move(path)) {
        VideoCapture cap(video_path);
        if (cap.isOpened()) {
            cout << "Video opened successfully" << endl;
            cap.release();
        } else {
            cout << "Error: Give a valid video path" << endl;
        }
    }

    // Public method to start processing
    void denoising_process() {
        denoising_video();
    }
    vector<ChunkData> chunking_video(int chunkDuration, int videoIndex) {
        return chunkVideo(video_path, chunkDuration, videoIndex, "myvideo");
    }

};

int main(int argc,char* argv[]) {
    string video_path;
    if (argc < 2){
        cout << "Give a valid video path" << endl;
        return 1;
    }
    else{
        video_path = argv[1];
    }
    Video video(video_path);
    vector<ChunkData> chunks;
    try {
        chunks = video.chunking_video(600,0); // Replace with your video path
        std::cout << "Video chunked successfully!" << std::endl;
        for (const auto& chunk : chunks) {
            std::cout << "Chunk " << chunk.chunkNum << ":" << std::endl;
            std::cout << "  Video: " << chunk.videoPath << std::endl;
            std::cout << "  Audio: " << chunk.audioPath << std::endl;
            Video cap(chunk.videoPath);
            cap.denoising_process();
        }
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
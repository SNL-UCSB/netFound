#include "PcapFileDevice.h"
#include "Packet.h"
#include "IPv4Layer.h"
#include <IPv6Layer.h>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3 or argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <inputfile> <outputfile> [optional: <unixtime_seconds to set as a start of the file>]\n";
        return 1;
    }

    auto *reader = pcpp::IFileReaderDevice::getReader(argv[1]);

    if (reader == nullptr) {
        std::cerr << "Cannot determine reader for file: " << argv[1] << std::endl;
        return 1;
    }
    if (!reader->open()) {
        std::cerr << "Cannot open " << argv[1] << " for reading" << std::endl;
        return 1;
    }


    std::string outFileName = argv[2];
    pcpp::IFileWriterDevice *writer;
    if (outFileName.ends_with(".pcap")) {
        writer = new pcpp::PcapFileWriterDevice(argv[2]);
    } else if (outFileName.ends_with(".pcapng")) {
        writer = new pcpp::PcapNgFileWriterDevice(argv[2]);
    } else {
        std::cerr << "Output file must have .pcap or .pcapng extension" << std::endl;
        return 1;
    }

    if (!writer->open()) {
        // Handle error
        std::cerr << "Error opening output file: " << argv[2] << std::endl;
        return 1;
    }

    // parse the optional argument
    bool enable_time_shift = false;
    long unixtime_seconds = 0;
    if (argc == 4) {
        unixtime_seconds = std::stol(argv[3]);
        enable_time_shift = true;
    }

    long diff = 0;
    bool first = true;

    pcpp::RawPacket rawPacket;
    while (reader->getNextPacket(rawPacket)) {
        if (enable_time_shift) {
            if (first) {
                first = false;
                diff = rawPacket.getPacketTimeStamp().tv_sec - unixtime_seconds;
            } else {
                auto x = rawPacket.getPacketTimeStamp();
                x.tv_sec -= diff;
                rawPacket.setPacketTimeStamp(x);
            }
        }

        pcpp::Packet parsedPacket(&rawPacket);
        int ipversion = 0;
        auto *ipLayer = parsedPacket.getLayerOfType<pcpp::IPv4Layer>();
        if (ipLayer != nullptr) {
            ipversion = 4;
        } else {
            auto *ipv6Layer = parsedPacket.getLayerOfType<pcpp::IPv6Layer>();
            if (ipv6Layer != nullptr) {
                ipversion = 6;
            }
        }
        if (ipversion != 0) {
            if (parsedPacket.isPacketOfType(pcpp::TCP) ||
                parsedPacket.isPacketOfType(pcpp::UDP) ||
                parsedPacket.isPacketOfType(pcpp::ICMP)) {
                writer->writePacket(rawPacket);
            }
        }
    }

    reader->close();
    writer->close();

    return 0;
}



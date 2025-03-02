#!/usr/bin/env python3
#
# profiler.py

#import matplotlib.pyplot as plt
#import numpy as np
import json
import sys
import getopt
import pyshark
import socket
import collections
import matplotlib.pyplot as plt
import numpy as np

dev = "en0"
pktCount = 1
fields = {}
fields["packets"] = 0

def help():
    print(__file__)
    print("\t-c\tNumber of packets to capture and process.")
    print("\t-d\tProfile a specific destination address.")
    print("\t-f\tPcap file you want to profile.")
    print("\t-h\tThis help output.")
    print("\t-i\tInterface name to listen to.")
    exit()

# Get options for execution
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "i:c:d:f:h:")

except:
    help()

if not opts:
    help()

for opt, arg in opts:
    if opt == '-i':
        dev = arg
    elif opt == '-c':
        pktCount = int(arg)
    elif opt == '-d':
        dstHost = arg
    elif opt == '-f':
        pcapFile = arg
    elif opt == '-h':
        help()

if 'dstHost' in locals():
    displayFilter = "ip.dst == " + dstHost
else:
    displayFilter = ""

def addOne(key, value):
    if key not in fields:
        fields[key] = {}

    if value in fields[key]:
        fields[key][value] +=1
    else:
        fields[key][value] = 1

"""
#Pre-populate headers.
def populate(key, crange):
    counter = 0
    if key not in fields:
        fields[key] = {}

    while counter <= crange:
        fields[key][counter] = 0
        counter += 1

populate("ipLenDist", 65535)
populate("ipTtlDist", 255)
populate("ipFlagsDist", 2)
populate("ipChecksumDist", 65535)
populate("icmpv6TypeDist", 255)
populate("icmpv6CodeDist", 255)
populate("icmpv6ChecksumDist", 65535)
populate("icmpTypeDist", 255)
populate("icmpCodeDist", 255)
populate("icmpChecksumDist", 65535)
populate("tcpWinSizeDist", 67107840)
populate("tcpSrcPortDist", 65535)
populate("tcpDestPortDist", 65535)
populate("tcpLengthDist", 65535)
populate("tcpChecksumDist", 65535)
populate("udpSrcPortDist", 65535)
populate("udpDestPortDist", 65535)
populate("udpLengthDist", 65535)
populate("udpChecksumDist", 65535)

fields["ipVersionDist"] = {} 
fields["ipVersionDist"]["4"] = 0
fields["ipVersionDist"]["6"] = 0

fields["ipProtoDist"] = {} 
fields["ipProtoDist"]["ARP"] = 0
fields["ipProtoDist"]["ICMPv6"] = 0
fields["ipProtoDist"]["ICMP"] = 0
fields["ipProtoDist"]["IGMP"] = 0
fields["ipProtoDist"]["TCP"] = 0
fields["ipProtoDist"]["UDP"] = 0
"""

if 'pcapFile' in locals():
    # file capture
    capture = pyshark.FileCapture(pcapFile, display_filter=displayFilter)
else:
    # live capture
    capture = pyshark.LiveCapture(interface=dev, display_filter=displayFilter)
    capture.sniff(packet_count=pktCount)


# Evaluate each packet
for p in capture:
    proto = "UNKNOWN"
    srcport = ""
    dstport = ""
    ipsrc = ""
    ipdst = ""

#    for i in dir(p):
#        print(i)
#
#    print(p.highest_layer)
#    print(p.transport_layer)
#    print(p.layers)

    # Determine layer 2 
    if 'arp' in p:
        proto = "ARP"
        ipsrc = p.arp.src_proto_ipv4
        ipdst = p.arp.dst_proto_ipv4

        addOne("ipSrcIpDist", p.arp.src_proto_ipv4)
        addOne("ipDstIpDist", p.arp.dst_proto_ipv4)

        #print(p.arp.field_names)

    # Deterimine layer 3 ipv4 vs ipv6
    if 'ip' in p:
        ipsrc = p.ip.src
        ipdst = p.ip.dst

        ipChecksum = int(p.ip.checksum, base=16)
        ipFlag = int(p.ip.flags, base=16)

        addOne("ipSrcIpDist", p.ip.src)
        addOne("ipDstIpDist", p.ip.dst)
        addOne("ipLenDist", p.ip.len)
        addOne("ipTtlDist", p.ip.ttl)
        addOne("ipVersionDist", p.ip.version)
        addOne("ipFlagsDist", ipFlag)
        addOne("ipChecksumDist", ipChecksum)

        #print(p.ip.field_names)

    if 'ipv6' in p:
        ipsrc = p.ipv6.src
        ipdst = p.ipv6.dst

        addOne("ipSrcIpDist", p.ipv6.src)
        addOne("ipDstIpDist", p.ipv6.dst)

        #print(p.ipv6.field_names)

    if 'icmpv6' in p:
        proto = "ICMPv6"
        icmpv6Checksum = int(p.icmpv6.checksum, base=16)

        addOne("icmpv6TypeDist", p.icmpv6.type)
        addOne("icmpv6CodeDist", p.icmpv6.code)
        addOne("icmpv6ChecksumDist", icmpv6Checksum)

        #print(p.icmpv6.field_names)
    
    if 'icmp' in p:
        proto = "ICMP"
        icmpChecksum = int(p.icmp.checksum, base=16)

        addOne("icmpTypeDist", p.icmp.type)
        addOne("icmpCodeDist", p.icmp.code)
        addOne("icmpChecksumDist", icmpChecksum)

        #print(p.icmp.field_names)

    if 'igmp' in p:
        proto = "IGMP"

        #print(p.igmp.field_names)

    # Determine layer 4 protocol
    if 'tcp' in p:
        proto = "TCP"
        tcpFlags = p.tcp.flags_str
        tcpFlags = tcpFlags.replace("\\xc2\\xb7", "")
        tcpFlags = tcpFlags.replace("·", "")
        tcpChecksum = int(p.tcp.checksum, base=16)

        addOne("tcpFlagDist", tcpFlags)
        addOne("tcpWinSizeDist", p.tcp.window_size)
        addOne("tcpSrcPortDist", p.tcp.srcport)
        addOne("tcpDestPortDist", p.tcp.dstport)
        addOne("tcpLengthDist", p.tcp.len)
        addOne("tcpChecksumDist", tcpChecksum)

        #print(p.tcp.field_names)

    if 'udp' in p:
        proto = "UDP"
        udpChecksum = int(p.udp.checksum, base=16)

        addOne("udpSrcPortDist", p.udp.srcport)
        addOne("udpDestPortDist", p.udp.dstport)
        addOne("udpLengthDist", p.udp.length)
        addOne("udpChecksumDist", udpChecksum)

        #print(p.udp.field_names)

    addOne("ipProtoDist", proto)
    fields["packets"] += 1

count = 0
mlist = []
imglist = []

fields = collections.OrderedDict(sorted(fields.items())) 

for d in fields:
    if d == "packets":
        continue
    else:
        for dt in fields[d]:
            newVal = fields[d][dt] / fields["packets"] * 100
            newVal = round(newVal, 2)
            fields[d][dt] = newVal

            mlist.append(newVal)
            count += 1

            if count == 50:
                count = 0
                imglist.append(mlist)
                del mlist
                mlist = []


output = json.dumps(fields, ensure_ascii=False, sort_keys=True, indent=4)
print(output)

plt.imshow(imglist, cmap='plasma', interpolation='hermite')
plt.axis('off')
plt.show()

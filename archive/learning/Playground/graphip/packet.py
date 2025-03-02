#!/usr/bin/env python3
#
# packet.py -i <interface> -c <packet_count> -h <neo4j_host> -p <neo4j_password> -u <neo4j_user>

from struct import *
import sys
import getopt
import socket
import pyshark
from graph import Graph

# Default settings
dev = "eth1"
pktCount = 1
neoUser = 'neo4j'
neoPass = '*****'
neoHost = '127.0.0.1'

# Get options for execution
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "i:c:h:p:u:")

except:
	print('packet.py -i <interface> -c <packet_count> -h <neo4j_host> -p <neo4j_password> -u <neo4j_user>')
	exit()

for opt, arg in opts:
    if opt == '-i':
        dev = arg
    elif opt == '-c':
        pktCount = int(arg)
    elif opt == '-h':
        neoHost = arg
    elif opt == '-p':
        neoPass = arg
    elif opt == '-u':
        neoUser = arg

# Prepare login items for neo4j
scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
port = 7687
url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=neoHost, port=port)

# Define IP Directory
ipDict = {}

# Init capture
capture = pyshark.LiveCapture(interface=dev)

# Evaluate each packet
for p in capture.sniff_continuously(packet_count=pktCount):

    proto = "UNKNOWN"
    srcport = ""
    dstport = ""

    # Determine layer 2 
    if 'arp' in p:
        proto = "ARP"
        ipsrc = p.arp.src_proto_ipv4
        ipdst = p.arp.dst_proto_ipv4
        #print(p.arp.field_names)

    # Deterimine layer 3 ipv4 vs ipv6
    if 'ip' in p:
        ipsrc = p.ip.src
        ipdst = p.ip.dst
        #print(p.ip.field_names)

    if 'ipv6' in p:
        ipsrc = p.ipv6.src
        ipdst = p.ipv6.dst
        #print(p.ipv6.field_names)

    if 'icmpv6' in p:
        proto = "ICMPv6"
        #print(p.icmpv6.field_names)
    
    if 'icmp' in p:
        proto = "ICMP"
        #print(p.icmp.field_names)

    if 'igmp' in p:
        proto = "IGMP"
        #print(p.igmp.field_names)

    # Determine layer 4 protocol
    if 'tcp' in p:
        proto = "TCP"
        srcport = p.tcp.srcport
        dstport = p.tcp.dstport
        #print(p.tcp.field_names)

    if 'udp' in p:
        proto = "UDP"
        srcport = p.udp.srcport
        dstport = p.udp.dstport
        #print(p.udp.field_names)

    if proto == "ARP" or proto == "UNKNOWN":
        pass
    else:
        sd = ipsrc + "-" + ipdst
        if sd in ipDict:
            ipDict[sd] +=1
        else:
            ipDict[sd] = 1

    if proto == "ARP" or proto == "ICMP" or proto == "ICMPv6" or proto == "IGMP":
        print('{} {} --> {}'.format(proto, ipsrc, ipdst))
    elif proto == "UNKNOWN":
        continue
    else:
        print('{} {}:{} --> {}:{}'.format(proto, ipsrc, srcport, ipdst, dstport))

graph = Graph(url, neoUser, neoPass)

counter = 0
for key in ipDict.keys():
    ipdat = key.split("-")
    graph.create_relation(ipdat[0], ipdat[1])
    counter += 1

graph.close()

print(f"{counter} unique relations created")


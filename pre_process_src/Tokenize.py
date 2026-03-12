import json
import logging
import os
from os.path import join
from multiprocessing import Pool
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import socket
from enum import Enum

import pyarrow as pa

from argparse import ArgumentParser, Namespace


class Protocol(Enum):
    TCP = 6
    UDP = 17
    ICMP = 1


PROTOCOL_FIELDS = {
    Protocol.TCP: "TCPFields",
    Protocol.UDP: "UDPFields",
    Protocol.ICMP: "ICMPFields",
}

FLOW_SIZE_LIMIT = 12
BURST_SIZE_LIMIT = 6
STRICT_IP_CHECK = False
MAX_CLASS_SIZE = 10000
BURST_SPLIT_BORDER = 10000000
INTERNAL_IPS = ["127.0.0.1/8"]
TOKEN_BYTES_LENGTH = 2
TCP_OPTIONS = False

## Reserved label names
FILENAME_SUBSTITUTION = "RESERVED_FILENAME"


def get_protocol(file_name: str) -> Optional[Protocol]:
    try:
        return Protocol(int(file_name.split(".")[-1]))
    except ValueError:
        return None


def tokenize_file(inpt_file, label) -> Optional[list]:
    try:
        bursts, flowDur = get_bursts_from_flow_file(inpt_file)

        protocol = get_protocol(inpt_file)
        if protocol is None:
            print(f"Error in file {inpt_file} : Protocol not found: {protocol}")
            return
        field_name = PROTOCOL_FIELDS.get(protocol)
        if field_name is None:
            print(
                f"Error in file {inpt_file} : fields are not defined for protocol {protocol}]"
            )
            return

        payloadTokenNum = config["Payload"][0]["numberTokens"]
        tokensPerPacket = sum([field["numberTokens"] for field in config["IPFields"]])
        tokensPerPacket += (
                sum([field["numberTokens"] for field in config[field_name]])
                + payloadTokenNum
        )

        bursts = bursts[
            bursts["burstID"] < FLOW_SIZE_LIMIT
            ]  # leave only first FLOW_SIZE_LIMIT bursts
        bursts = (
            bursts.groupby("burstID").head(BURST_SIZE_LIMIT).reset_index(drop=True)
        )  # leave only first BURST_SIZE_LIMIT packets in each burst

        grouped_bursts = bursts.groupby("burstID")
        grouped_bursts_stat = grouped_bursts.agg(
            {
                "IP_tl": "sum",
                "IAT": lambda x: int(round((x.iloc[0] / 1000))),
                "direction": "first",
            }
        ).sort_index()
        group_sizes = grouped_bursts.size().reset_index(name="packet_count")
        grouped_bursts_stat = grouped_bursts_stat.reset_index().merge(
            group_sizes, on="burstID", how="left"
        )
        grouped_bursts_stat["total_tokens"] = (
                tokensPerPacket * grouped_bursts_stat["packet_count"]
        )

        number_of_bursts = len(grouped_bursts_stat)
        directionsls = [
            grouped_bursts_stat["direction"][i] for i in range(number_of_bursts)
        ]
        bytels = [grouped_bursts_stat["IP_tl"][i] for i in range(number_of_bursts)]
        iatls = [grouped_bursts_stat["IAT"][i] for i in range(number_of_bursts)]
        countls = [
            grouped_bursts_stat["packet_count"][i] for i in range(number_of_bursts)
        ]

        tokenize_fields_df(config, "IPFields", bursts)
        tokenize_fields_df(config, field_name, bursts)
        tokenize_fields_df(config, "Payload", bursts)

        columns = (
                [x["field"] for x in config["IPFields"]]
                + [x["field"] for x in config[field_name]]
                + [x["field"] for x in config["Payload"]]
        )

        bursts.loc[:, "concatenated"] = bursts[columns].sum(axis=1)
        grouped_bursts = bursts.groupby("burstID")["concatenated"].sum()

        if len(grouped_bursts) > 0:
            return [
                flowDur,
                grouped_bursts,
                directionsls,
                bytels,
                iatls,
                countls,
                protocol.value,
                label,
            ]
    except Exception as e:
        print(f"Error in file {inpt_file} : {str(e)}")


def get_int_from_byte(byte_vals: bytes, byte_order="little"):
    if byte_vals == b"":
        raise ValueError("byte_vals is empty")
    # noinspection PyTypeChecker
    return int.from_bytes(byte_vals, byteorder=byte_order)


def get_bursts_from_flow_file(inpt_file):
    flow_rows = []
    protocol = get_protocol(inpt_file)

    with open(inpt_file, mode="rb") as f:
        assert protocol.value == get_int_from_byte(f.read(1))
        try:
            while True:
                ts = f.read(8)
                if ts == b"":
                    break  # end of file
                currFlowRow = [
                    get_int_from_byte(ts),
                    get_int_from_byte(f.read(1)),
                    get_int_from_byte(f.read(1)),
                    get_int_from_byte(f.read(2)),
                    get_int_from_byte(f.read(1)),
                    get_int_from_byte(f.read(1)),
                    get_int_from_byte(f.read(4)),
                    get_int_from_byte(f.read(4)),
                ]
                if protocol == Protocol.TCP:
                    f.read(2)  # srcport
                    f.read(2)  # dstport
                    currFlowRow.extend(
                        [
                            get_int_from_byte(f.read(1)),
                            get_int_from_byte(f.read(2)),
                            get_int_from_byte(f.read(4)),
                            get_int_from_byte(f.read(4)),
                            get_int_from_byte(f.read(2)),
                        ]
                    )
                    if TCP_OPTIONS:
                        currFlowRow.extend([get_int_from_byte(f.read(40), byte_order="big")])
                if protocol == Protocol.UDP:
                    f.read(2)  # srcport
                    f.read(2)  # dstport
                    currFlowRow.append(get_int_from_byte(f.read(2)))
                if protocol == Protocol.ICMP:
                    currFlowRow.extend(
                        [
                            get_int_from_byte(f.read(1)),
                            get_int_from_byte(f.read(1)),
                        ]
                    )
                currFlowRow.append(get_int_from_byte(f.read(12), byte_order="big"))
                flow_rows.append(currFlowRow)
        except ValueError:
            print(f"Unexpected end of file occured for {inpt_file}")
            return None, 0

    if len(flow_rows) == 0:
        return None, 0

    columns = [
        "rts",
        "IP_hl",
        "IP_tos",
        "IP_tl",
        "IP_Flags",
        "IP_ttl",
        "SrcIP",
        "DstIP",
    ]

    if protocol == Protocol.TCP:
        columns += [
            "TCP_Flags",
            "TCP_wsize",
            "TCP_seq",
            "TCP_ackn",
            "TCP_urp",
        ]
        if TCP_OPTIONS:
            columns.append("TCP_options")
        columns.append("Payload")
    elif protocol == Protocol.UDP:
        columns += [
            "UDP_len",
            "Payload",
        ]
    else:
        columns += [
            "ICMP_type",
            "ICMP_code",
            "Payload",
        ]

    df = pd.DataFrame(flow_rows, columns=columns)

    df["rts"] = df["rts"].astype(int)
    df["rts"] -= df["rts"].min()
    flowDur = int((df["rts"].max() / 1000))
    df = df.sort_values(by="rts", ignore_index=True, kind="stable")
    fwdDf, bkdDf = split_bursts_on_dir(df, inpt_file)
    if fwdDf is None and bkdDf is None:
        fwdDf = df
    combinedBurstLen = (0 if fwdDf is None else fwdDf.shape[0]) + (
        0 if bkdDf is None else bkdDf.shape[0]
    )
    if df.shape[0] > combinedBurstLen:
        print(
            f"Original length : {df.shape[0]} but combined length : {combinedBurstLen}"
        )
    fwdBursts = split_based_on_iat(fwdDf, starting_index=0)
    fwdBursts["direction"] = True
    starting_index = (
        0
        if (fwdBursts is None or fwdBursts.shape[0] == 0)
        else fwdBursts.burstID.max() + 1
    )
    bkdBursts = split_based_on_iat(bkdDf, starting_index=starting_index)
    bkdBursts["direction"] = False
    bursts = (
        (pd.concat([fwdBursts, bkdBursts]) if bkdBursts is not None else fwdBursts)
        .sort_values(by="first_packet_time", ignore_index=True, kind="stable")
        .drop(columns=["first_packet_time"])
    )
    bursts["burstID"] = bursts["burstID"].diff().ne(0).cumsum() - 1

    if bursts.shape[0] == 0:
        return None, 0
    if bursts is None:
        logger.info(f"No burst in file {inpt_file}")
        return None, 0

    return bursts, flowDur


def split_bursts_on_dir(df, inputfile):
    ipSet = set(df["SrcIP"]).union(set(df["DstIP"]))

    if len(ipSet) > 2:
        ipsToPrint = ",".join([int_to_ip_address(ip) for ip in ipSet])
        print(f"inputfile {inputfile} has flows IPs {ipsToPrint}")
        raise ValueError(f"inputfile {inputfile} has flows IPs {ipsToPrint}")
    srcIP = None
    for ip in ipSet:
        if is_internal_ip(ip):
            srcIP = ip
            break
    if srcIP is None:
        ipList = ",".join([int_to_ip_address(ip) for ip in ipSet])
        if STRICT_IP_CHECK:
            logger.error(f"IPs {ipList} not in internalIPs for {inputfile}")
            return None, None
        srcIP = df.SrcIP[0]
    retBkd = df[df.SrcIP != srcIP]
    retFwd = df[df.SrcIP == srcIP]
    return retFwd, retBkd


def convert_ip_str_to_bits(ip_with_subnet):
    split_vals = ip_with_subnet.split("/")
    ip = split_vals[0]
    if len(split_vals) == 1:
        subnetRange = "32"
    else:
        subnetRange = ip_with_subnet.split("/")[1]
    return (
            "".join([bin(int(octet)).split("b")[1].zfill(8) for octet in ip.split(".")])
            + "/"
            + subnetRange
    )


def int_to_ip_address(ip_int):
    # Convert integer to IP address
    return socket.inet_ntoa(int.to_bytes(ip_int, 4, "big"))


def is_ip_in_range(ip, ip_range):
    subnetLength = int(ip_range.split("/")[1])
    return (
            ip[:subnetLength]
            == convert_ip_str_to_bits(ip_range.split("/")[0])[:subnetLength]
    )


def is_internal_ip(ip):
    ip = "{0:b}".format(ip).rjust(32, "0")
    return any([is_ip_in_range(ip, ipRange) for ipRange in INTERNAL_IPS])


def split_based_on_iat(
        df: pd.DataFrame, starting_index: int = 0
) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.reset_index(drop=True)

    burstStartIdx = df[
        (df.rts - df.rts.shift(1, axis=0, fill_value=(-1 * BURST_SPLIT_BORDER - 1)))
        > BURST_SPLIT_BORDER
        ].index
    if len(burstStartIdx) == 0:
        df["IAT"] = 0
        df["burstID"] = 0
        return df

    df["burstID"] = df.index.isin(burstStartIdx).cumsum() + starting_index

    first_packets = df.groupby("burstID")["rts"].first().reset_index()
    first_packets.rename(columns={"rts": "first_packet_time"}, inplace=True)

    first_packets["IAT"] = first_packets["first_packet_time"].diff().fillna(0)
    df = df.merge(
        first_packets[["burstID", "first_packet_time", "IAT"]], on="burstID", how="left"
    )
    df.loc[df["burstID"] == starting_index + 1, "IAT"] = 0
    df["IAT"] = df["IAT"].astype(int)
    return df


def tokenize_fields_df(_config, type_of_field, df):
    for ipConf in _config[type_of_field]:
        field = ipConf["field"]
        numberOfTokens = ipConf["numberTokens"]
        df[field] = df[field].apply(
            lambda val: val.to_bytes(
                numberOfTokens * TOKEN_BYTES_LENGTH, byteorder="big"
            )
        )


def get_logger(logger_file: str) -> logging.Logger:
    _logger = logging.getLogger("TokenizerLog")
    _logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logger_file)
    fh.setLevel(logging.DEBUG)
    _logger.addHandler(fh)
    return _logger


def slice_bytes_to_16bit_tokens(burst_tokens: pd.Series) -> list[list[int]]:
    result = burst_tokens.apply(
            lambda x: [
                int.from_bytes(x[i: i + TOKEN_BYTES_LENGTH], byteorder="big")
                for i in range(0, len(x), TOKEN_BYTES_LENGTH)
            ]
        )

    return result


def tokenizer_helper(
        output_filename: str,
        tokenization_args: List[Tuple[str, Optional[str]]],
        batch_size: Optional[int] = 1000,
) -> None:
    flow_duration_type = pa.uint64()
    burst_tokens_type = pa.list_(pa.list_(pa.uint16()))
    directions_type = pa.list_(pa.bool_())
    bytes_type = pa.list_(pa.uint32())
    iats_type = pa.list_(pa.uint64())
    counts_type = pa.list_(pa.uint32())
    protocol_type = pa.uint16()
    label_type = pa.string()

    table_schema = pa.schema(
        [
            pa.field("flow_duration", flow_duration_type),
            pa.field("burst_tokens", burst_tokens_type),
            pa.field("directions", directions_type),
            pa.field("bytes", bytes_type),
            pa.field("iats", iats_type),
            pa.field("counts", counts_type),
            pa.field("protocol", protocol_type),
            pa.field("labels", label_type),
        ]
    )

    # flow_durations, burst_tokens, directions, bytes_ar, iats, counts, protocols, labels
    data = [[] for _ in range(8)]

    with pa.OSFile(output_filename, "wb") as sink:
        with pa.ipc.new_stream(sink, schema=table_schema) as writer:
            total_files = len(tokenization_args)
            for i, (inpt_file, label) in enumerate(tokenization_args, start=1):
                result = tokenize_file(inpt_file, label)

                if result is not None:
                    result[1] = slice_bytes_to_16bit_tokens(result[1])
                    for j in range(8):
                        data[j].append(result[j])

                if i % batch_size == 0 or i == total_files:
                    print(f"Processed {i} files")
                    if data[1]:
                        batch = pa.record_batch(
                            [
                                pa.array(data[0], type=flow_duration_type),
                                pa.array(data[1], type=burst_tokens_type),
                                pa.array(data[2], type=directions_type),
                                pa.array(data[3], type=bytes_type),
                                pa.array(data[4], type=iats_type),
                                pa.array(data[5], type=counts_type),
                                pa.array(data[6], type=protocol_type),
                                pa.array(data[7], type=label_type),
                            ],
                            schema=table_schema,
                        )
                        writer.write_batch(batch)
                        data = [[] for _ in range(8)]


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--conf_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--logger_file", type=str, default="/tmp/tokenizer.log")
    parser.add_argument("--flow_size_limit", type=int, default=12)
    parser.add_argument("--burst_size_limit", type=int, default=6)
    parser.add_argument("--burst_split_border", type=float, default=10000000)
    parser.add_argument("--strict_ip_checking", type=bool, default=False)
    parser.add_argument("--max_class_size", type=int, default=10000)
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument("--arrow_batch_size", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    script_args = get_args()

    FLOW_SIZE_LIMIT = script_args.flow_size_limit
    BURST_SIZE_LIMIT = script_args.burst_size_limit
    BURST_SPLIT_BORDER = script_args.burst_split_border
    STRICT_IP_CHECK = script_args.strict_ip_checking
    MAX_CLASS_SIZE = script_args.max_class_size

    logger = get_logger(script_args.logger_file)

    with open(script_args.conf_file, "r") as config_file:
        config = json.load(config_file)

    INTERNAL_IPS = config["internalIPs"]

    for field in config["TCPFields"]:
        if field["field"] == "TCP_options":
            TCP_OPTIONS = True

    if script_args.input_dir:
        input_dir = script_args.input_dir
    else:
        input_dir = config["input_dir"]

    if script_args.output_dir:
        output_dir = script_args.output_dir
    else:
        output_dir = config["output_dir"]

    args = [
        (
            join(input_dir, file),
            script_args.label if script_args.label != FILENAME_SUBSTITUTION else join(input_dir, file),
        ) 
        for file 
        in os.listdir(input_dir)
    ]
    
    # validation for tcp options
    tcpoptions_pattern = f"tcpoptions.{Protocol.TCP.value}"
    if TCP_OPTIONS:
        for filepath, _ in args:
            if filepath.endswith(f".{Protocol.TCP.value}") and not filepath.endswith(tcpoptions_pattern):
                raise ValueError(f"TCP options are enabled, but file {filepath} does not have tcp options")
    else:
        for filepath, _ in args:
            if filepath.endswith(tcpoptions_pattern):
                raise ValueError(f"TCP options are disabled, but file {filepath} has tcp options")

    print(f"Total files: {len(args)}")

    cores = script_args.cores
    if cores == 0:
        cores = min(os.cpu_count() - 2, len(args))

    # split args to cores equal lists
    args = np.array_split(args, cores)
    input_args = (
        (join(output_dir, f"shard.{i}.arrow"), args[i], script_args.arrow_batch_size)
        for i in range(cores)
    )

    print(f"Started processing files, time: {pd.Timestamp.now()}")
    with Pool(cores) as p:
        p.starmap(tokenizer_helper, input_args)
    print(f"Finished processing files, time: {pd.Timestamp.now()}")

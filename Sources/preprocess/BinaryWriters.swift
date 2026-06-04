import Domain
import Foundation

func writeIVF(
    path: String,
    count: Int,
    stride: Int,
    clusterCount: Int,
    centroids: [Int16],
    bboxMin: [Int16],
    bboxMax: [Int16],
    offsets: [UInt32],
    postings: [UInt32],
    orderedVectors: [Int16],
    orderedLabels: [UInt8],
    createdUnix: Int64
) throws {
    var output = Data()
    output.append(contentsOf: [0x52, 0x49, 0x56, 0x46]) // "RIVF"
    output.appendLE(UInt32(3))
    output.appendLE(UInt64(count))
    output.appendLE(UInt32(clusterCount))
    output.appendLE(UInt32(stride))
    output.appendLE(createdUnix)
    output.padTo(alignment: IVFHeader.bytes)

    centroids.withUnsafeBufferPointer { buffer in
        let byteCount = centroids.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    bboxMin.withUnsafeBufferPointer { buffer in
        let byteCount = bboxMin.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    bboxMax.withUnsafeBufferPointer { buffer in
        let byteCount = bboxMax.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    offsets.withUnsafeBufferPointer { buffer in
        let byteCount = offsets.count * MemoryLayout<UInt32>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    postings.withUnsafeBufferPointer { buffer in
        let byteCount = postings.count * MemoryLayout<UInt32>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    orderedVectors.withUnsafeBufferPointer { buffer in
        let byteCount = orderedVectors.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    orderedLabels.withUnsafeBufferPointer { buffer in
        output.append(buffer.baseAddress!, count: orderedLabels.count)
    }

    try output.write(to: URL(fileURLWithPath: path))
}

func writePartition(
    path: String,
    count: Int,
    stride: Int,
    dim: Int,
    offsets: [UInt32],
    orderedVectors: [Int16],
    orderedLabels: [UInt8]
) throws {
    var output = Data()
    output.append(contentsOf: [0x52, 0x50, 0x4B, 0x59]) // "RPKY"
    output.appendLE(UInt32(1))                            // version
    output.appendLE(UInt64(count))
    output.appendLE(UInt32(stride))
    output.appendLE(UInt32(dim))
    output.padTo(alignment: 64)                           // header = 64 bytes

    // 257 offsets
    offsets.withUnsafeBufferPointer { buffer in
        let byteCount = offsets.count * MemoryLayout<UInt32>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: 4096)

    orderedVectors.withUnsafeBufferPointer { buffer in
        let byteCount = orderedVectors.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: 4096)

    orderedLabels.withUnsafeBufferPointer { buffer in
        output.append(buffer.baseAddress!, count: orderedLabels.count)
    }

    try output.write(to: URL(fileURLWithPath: path))
}

func writeIVFPQ(
    path: String,
    count: Int,
    stride: Int,
    subvectorCount: Int,
    subvectorWidth: Int,
    codebooks: [Int16],
    codes: [UInt8],
    createdUnix: Int64
) throws {
    var output = Data()
    output.append(contentsOf: [0x52, 0x56, 0x51, 0x50]) // "RVQP"
    output.appendLE(UInt32(1))
    output.appendLE(UInt64(count))
    output.appendLE(UInt32(stride))
    output.appendLE(UInt32(subvectorCount))
    output.appendLE(UInt32(subvectorWidth))
    output.appendLE(UInt32(0))
    output.appendLE(createdUnix)
    output.padTo(alignment: IVFPQHeader.bytes)

    codebooks.withUnsafeBufferPointer { buffer in
        let byteCount = codebooks.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFPQHeader.pageAlignment)

    codes.withUnsafeBufferPointer { buffer in
        output.append(buffer.baseAddress!, count: codes.count)
    }

    try output.write(to: URL(fileURLWithPath: path))
}

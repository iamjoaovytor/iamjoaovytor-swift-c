import Domain

func buildPartition(
    lanes: [Int16],
    labels: [UInt8],
    count: Int,
    stride: Int
) -> (offsets: [UInt32], orderedVectors: [Int16], orderedLabels: [UInt8]) {
    // Assign each reference to its partition
    var partitionOf = [UInt8](repeating: 0, count: count)
    var partitionCounts = [Int](repeating: 0, count: PartitionKey.count)

    for i in 0..<count {
        let base = i * stride
        let key = lanes.withUnsafeBufferPointer { buf in
            PartitionKey.compute(for: buf.baseAddress!.advanced(by: base))
        }
        partitionOf[i] = key
        partitionCounts[Int(key)] += 1
    }

    // Build offsets (257 entries, last = count)
    var offsets = [UInt32](repeating: 0, count: PartitionKey.count + 1)
    var running = 0
    for p in 0..<PartitionKey.count {
        offsets[p] = UInt32(running)
        running += partitionCounts[p]
    }
    offsets[PartitionKey.count] = UInt32(count)

    // Fill ordered vectors and labels
    var orderedVectors = [Int16](repeating: 0, count: count * stride)
    var orderedLabels  = [UInt8](repeating: 0, count: count)
    var writePos = [Int](repeating: 0, count: PartitionKey.count)
    for p in 0..<PartitionKey.count {
        writePos[p] = Int(offsets[p])
    }

    for i in 0..<count {
        let pk = Int(partitionOf[i])
        let dst = writePos[pk]
        let src = i * stride
        for lane in 0..<stride {
            orderedVectors[dst * stride + lane] = lanes[src + lane]
        }
        orderedLabels[dst] = labels[i]
        writePos[pk] += 1
    }

    return (offsets, orderedVectors, orderedLabels)
}

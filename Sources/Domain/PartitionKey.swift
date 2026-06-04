public enum PartitionKey {
    public static let count = 256

    @inline(__always)
    public static func compute(for vector: UnsafePointer<Int16>) -> UInt8 {
        var key: UInt8 = 0
        if vector[5] >= 0     { key |= 0x01 }  // has_last_tx
        if vector[9] > 0      { key |= 0x02 }  // is_online
        if vector[10] > 0     { key |= 0x04 }  // card_present
        if vector[11] > 0     { key |= 0x08 }  // unknown_merchant
        if vector[12] >= 5000 { key |= 0x10 }  // high_mcc_risk (>= 0.5 * scale)
        if vector[2] >= 10000 { key |= 0x20 }  // amount > avg (>= 1.0 * scale)
        if vector[8] >= 5000  { key |= 0x40 }  // high_tx_count (>= 0.5 * scale)
        if vector[7] >= 5000  { key |= 0x80 }  // far_from_home (>= 0.5 * scale)
        return key
    }

    @inline(__always)
    public static func compute(for vector: [Int16]) -> UInt8 {
        vector.withUnsafeBufferPointer { buf in
            compute(for: buf.baseAddress!)
        }
    }
}

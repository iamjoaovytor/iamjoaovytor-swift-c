extension KNN {
    /// Returns true when the query has unambiguous risk signal on all key dimensions,
    /// allowing nprobe=1 for the initial IVF probe instead of initialNprobe.
    ///
    /// Thresholds are semantic (not calibrated against the official dataset):
    ///   scale=10000, maxAmount=10_000, maxKm=1_000, maxTxCount24h=20, maxInstallments=12
    ///
    /// dim layout (Vectorizer.swift):
    ///   [0] amount/maxAmount  [1] installments/12  [7] kmFromHome/1000
    ///   [8] txCount24h/20     [11] unknownMerchant (0=known, 10000=unknown)
    ///   [12] mccRisk (0..10000)
    @inline(__always)
    static func isObvious(query: [Int16]) -> Bool {
        // obvious legit: low amount, few installments, close to home,
        // few txs, known merchant, safe MCC — all conditions must hold
        if query[0] <= 500 &&   // amount ≤ $500
           query[1] <= 2500 &&  // installments ≤ 3
           query[7] <= 500 &&   // km_from_home ≤ 50 km
           query[8] <= 2500 &&  // tx_count_24h ≤ 5
           query[11] == 0 &&    // known merchant
           query[12] <= 3000    // mcc_risk ≤ 0.30
        { return true }

        // obvious fraud: high amount, many installments, far from home,
        // many txs, unknown merchant, risky MCC — all conditions must hold
        if query[0] >= 5000 &&   // amount ≥ $5000
           query[1] >= 4167 &&   // installments ≥ 5
           query[7] >= 1500 &&   // km_from_home ≥ 150 km
           query[8] >= 3000 &&   // tx_count_24h ≥ 6
           query[11] == 10000 && // unknown merchant
           query[12] >= 7500     // mcc_risk ≥ 0.75
        { return true }

        // ultra-fraud: extreme amount, unknown merchant, very risky MCC, very far from home
        if query[0] >= 8000 &&   // amount ≥ $8000
           query[7] >= 2000 &&   // km_from_home ≥ 200 km
           query[11] == 10000 && // unknown merchant
           query[12] >= 9000     // mcc_risk ≥ 0.90
        { return true }

        return false
    }
}

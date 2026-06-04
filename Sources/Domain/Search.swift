import CSearch

public enum KNN {
    /// Exact k-NN against the `references.bin` mapping using a native C
    /// kernel. On x86_64 the kernel switches to AVX2 when supported at
    /// runtime; other architectures stay on the scalar C path.
    public static func topK(
        query: [Int16],
        in index: ReferencesIndex,
        k: Int = 5
    ) -> [Neighbor] {
        withTopKRaw(query: query, in: index, k: k) { rawNeighbors in
            rawNeighbors.map {
                Neighbor(
                    recordIndex: Int($0.record_index),
                    distanceSquared: $0.distance_squared
                )
            }
        }
    }

    public static func fraudVoteCount(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex? = nil,
        pq: IVFPQIndex? = nil,
        pkey: PartitionIndex? = nil,
        config: SearchConfig = SearchConfig(),
        metrics: UnsafeMutablePointer<SearchMetrics>? = nil,
        k: Int = 5
    ) -> Int {
        if let pkey {
            let pk = PartitionKey.compute(for: query)
            let (votes, worstDist) = pkey.fraudVoteCount(
                query: query,
                partitionKey: pk,
                k: k
            )
            // If top-k is filled and result is non-ambiguous, trust the partition
            let isUnambiguous = votes == 0 || votes == k
            if isUnambiguous && worstDist < Int64.max {
                return votes
            }
            // Fall through to IVF with worstDist hint
            if let ivf {
                return fraudVoteCountIVF(
                    query: query,
                    in: index,
                    ivf: ivf,
                    pq: pq,
                    config: config,
                    metrics: metrics,
                    k: k
                )
            }
        }
        if let ivf {
            return fraudVoteCountIVF(
                query: query,
                in: index,
                ivf: ivf,
                pq: pq,
                config: config,
                metrics: metrics,
                k: k
            )
        }
        return fraudVoteCountExact(query: query, in: index, k: k)
    }
}

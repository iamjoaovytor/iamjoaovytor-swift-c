import Foundation

public struct SearchConfig: Sendable {
    public let nprobe: Int
    public let initialNprobe: Int
    public let adaptiveMinFraudVotes: Int
    public let adaptiveMaxFraudVotes: Int
    public let ivfpqRerankCandidates: Int?
    public let useBoundingBoxes: Bool
    public let expandOnUnanimousInitialVotes: Bool
    public let useBoundingBoxesOnExpandedSearch: Bool

    public var adaptiveEnabled: Bool {
        initialNprobe < nprobe && adaptiveMinFraudVotes <= adaptiveMaxFraudVotes
    }

    public init(
        nprobe: Int = 8,
        initialNprobe: Int? = nil,
        adaptiveMinFraudVotes: Int = 2,
        adaptiveMaxFraudVotes: Int = 3,
        ivfpqRerankCandidates: Int? = nil,
        useBoundingBoxes: Bool = false,
        expandOnUnanimousInitialVotes: Bool = false,
        useBoundingBoxesOnExpandedSearch: Bool = false
    ) {
        let clampedNProbe = max(1, nprobe)
        let clampedInitial = min(max(1, initialNprobe ?? clampedNProbe), clampedNProbe)
        let clampedRerank = ivfpqRerankCandidates.map { max(1, $0) }
        self.nprobe = clampedNProbe
        self.initialNprobe = clampedInitial
        self.adaptiveMinFraudVotes = adaptiveMinFraudVotes
        self.adaptiveMaxFraudVotes = adaptiveMaxFraudVotes
        self.ivfpqRerankCandidates = clampedRerank
        self.useBoundingBoxes = useBoundingBoxes
        self.expandOnUnanimousInitialVotes = expandOnUnanimousInitialVotes
        self.useBoundingBoxesOnExpandedSearch = useBoundingBoxesOnExpandedSearch
    }

    public func shouldExpand(after fraudVotes: Int) -> Bool {
        guard adaptiveEnabled else {
            return false
        }
        if fraudVotes >= adaptiveMinFraudVotes &&
            fraudVotes <= adaptiveMaxFraudVotes {
            return true
        }
        if expandOnUnanimousInitialVotes {
            return fraudVotes == 0 || fraudVotes == 5
        }
        return false
    }

    public func expandedSearchUsesBoundingBoxes() -> Bool {
        useBoundingBoxes || useBoundingBoxesOnExpandedSearch
    }

    // If the k-th nearest neighbor's distance² is ≤ this value, the result is
    // confident enough to stop scanning further clusters. Derived semantically:
    // 0.14 normalized distance → 1400 quantized (scale=10000) → 1400² = 1_960_000.
    public static let earlyDistanceLimit: Int64 = 1_960_000

    public var ivfpqEnabled: Bool {
        if let ivfpqRerankCandidates {
            return ivfpqRerankCandidates > 5
        }
        return false
    }
}

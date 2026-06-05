import Foundation

public struct SearchConfig: Sendable {
    public let nprobe: Int
    public let initialNprobe: Int
    public let midNprobe: Int
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
        midNprobe: Int? = nil,
        adaptiveMinFraudVotes: Int = 2,
        adaptiveMaxFraudVotes: Int = 3,
        ivfpqRerankCandidates: Int? = nil,
        useBoundingBoxes: Bool = false,
        expandOnUnanimousInitialVotes: Bool = false,
        useBoundingBoxesOnExpandedSearch: Bool = false,
        voteEarlyExitLimit: Int64 = 0
    ) {
        let clampedNProbe = max(1, nprobe)
        let clampedInitial = min(max(1, initialNprobe ?? clampedNProbe), clampedNProbe)
        let clampedMid = midNprobe.map { min(max(clampedInitial, $0), clampedNProbe) } ?? clampedNProbe
        let clampedRerank = ivfpqRerankCandidates.map { max(1, $0) }
        self.nprobe = clampedNProbe
        self.initialNprobe = clampedInitial
        self.midNprobe = clampedMid
        self.adaptiveMinFraudVotes = adaptiveMinFraudVotes
        self.adaptiveMaxFraudVotes = adaptiveMaxFraudVotes
        self.ivfpqRerankCandidates = clampedRerank
        self.useBoundingBoxes = useBoundingBoxes
        self.expandOnUnanimousInitialVotes = expandOnUnanimousInitialVotes
        self.useBoundingBoxesOnExpandedSearch = useBoundingBoxesOnExpandedSearch
        self.voteEarlyExitLimit = max(0, voteEarlyExitLimit)
    }

    // Returns the next nprobe to use given current votes and current probe count,
    // or nil if the result is accepted as-is.
    public func nextProbeCount(votes: Int, currentNprobe: Int) -> Int? {
        guard adaptiveEnabled, currentNprobe < nprobe else { return nil }

        let isAmbiguous = votes >= adaptiveMinFraudVotes && votes <= adaptiveMaxFraudVotes
        if isAmbiguous {
            return nprobe
        }

        let isUnanimous = votes == 0 || votes == 5
        if isUnanimous && expandOnUnanimousInitialVotes && currentNprobe < midNprobe {
            return midNprobe
        }

        return nil
    }

    public func expandedSearchUsesBoundingBoxes() -> Bool {
        useBoundingBoxes || useBoundingBoxesOnExpandedSearch
    }

    // If the k-th nearest neighbor's distance² is ≤ this value, the result is
    // confident enough to stop scanning further clusters. Derived semantically:
    // 0.14 normalized distance → 1400 quantized (scale=10000) → 1400² = 1_960_000.
    public static let earlyDistanceLimit: Int64 = 1_960_000

    // If votes are unanimous (0 or k) AND k-th distance² is ≤ this value, stop
    // scanning further clusters. Configurable via IVF_VOTE_EARLY_EXIT_LIMIT env var.
    // Default 0 disables the check; set to e.g. 2_500_000 (dist≤0.158) to enable.
    public let voteEarlyExitLimit: Int64

    public var ivfpqEnabled: Bool {
        if let ivfpqRerankCandidates {
            return ivfpqRerankCandidates > 5
        }
        return false
    }
}

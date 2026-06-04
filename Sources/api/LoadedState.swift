import Domain

struct LoadedState: Sendable {
    let index: ReferencesIndex
    let ivf: IVFIndex?
    let pq: IVFPQIndex?
    let pkey: PartitionIndex?
    let searchConfig: SearchConfig
    let vectorizer: Vectorizer
}

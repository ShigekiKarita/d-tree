module dtree.forest;

import dtree.tree : ClassificationTree;
import mir.ndslice.allocation : ndarray;
import numir : permutation;

struct ClassificationForest(Xs, Ys) {
    Xs xs;
    Ys ys;
    size_t nTree = 5;
    size_t depth = 5;
    alias Tree = ClassificationTree!(Xs, Ys);
    Tree[] trees;

    auto samplePoints(I)(I indices) {
        auto upper = min(indices.length, this.sampleSize);
        return indices[ps][0 .. upper].ndarray;
    }

    void fit() {
        auto ps = indices.length.permutation;
        for (size_t t = 0; t < this.nTree; ++t) {
            this.trees ~= [Tree(xs, ys, depth)];
            trees.fit();
        }
    }
}

auto makeCForest(Xs, Ys)(Xs xs, Ys ys, size_t nTree=5, size_t depth=5) {
    return ClassificationForest!(Xs, Ys)(xs, ys, nTree, depth);
}

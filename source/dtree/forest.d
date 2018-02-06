module dtree.forest;

import dtree.tree : ClassificationTree;
import mir.ndslice : iota, ndarray;
import numir : permutation, zeros;

struct ClassificationForest {
    // TODO make Tree argument object
    size_t nClass = 2;
    size_t depth = 5;

    size_t nTree = 5;

    ClassificationTree[] trees;

    auto samplePoints(I)(I indices) {
        auto upper = min(indices.length, this.sampleSize);
        return indices[ps][0 .. upper].ndarray;
    }

    void fit(Xs, Ys)(Xs xs, Ys ys) {
        import std.math : ceil;
        import std.algorithm : min;
        import std.conv : to;
        auto ps = ys.length.permutation;
        const stride = ceil(ys.length.to!double / this.nTree).to!size_t;
        for (size_t t = 0; t < this.nTree; ++t) {
            auto tree = ClassificationTree(this.nClass, this.depth);
            auto a = t * stride;
            auto b = min(a + stride, ps.length);
            auto ab = ps[a .. b];
            tree.fit(xs[ab], ys[ab]);
            this.trees ~= [tree];
        }
    }

    auto predict(X)(X x) {
        import mir.ndslice : map, iota, sliced;
        import mir.ndslice.allocation : ndarray;
        import mir.math : sum;
        auto result = zeros(this.nClass);
        foreach (t; this.trees) {
            result[] += t.predict(x).sliced;
        }
        result[] /= this.nTree;
        return result.ndarray;
    }
}

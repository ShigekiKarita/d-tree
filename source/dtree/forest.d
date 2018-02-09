module dtree.forest;

// import dtree.tree : ClassificationTree;
import mir.ndslice : iota, ndarray;
import numir : permutation, zeros;
import dtree.impurity : entropy;


struct RandomForest(Tree) {
    Tree initTree;
    size_t nTree = 5;
    bool bootstrap = true;
    Tree[] trees;

    void fit(Xs, Ys)(Xs xs, Ys ys) in {
        assert(xs.length == ys.length);
    } body {
        // TODO implement boot strap sampling
        // 1. sampling sample-id from multinomial dist
        // 2. sampling feature-id fromm multinomial dist
        import std.math : ceil;
        import std.algorithm : min;
        import std.conv : to;
        auto ps = ys.length.permutation;
        const stride = ys.length / this.nTree;
        for (size_t t = 0; t < this.nTree; ++t) {
            auto tree = initTree;
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
        auto result = zeros(this.initTree.nOutput);
        foreach (t; this.trees) {
            result[] += t.predict(x).sliced;
        }
        result[] /= this.nTree; // result.sum!"fast";
        return result.ndarray;
    }
}


auto toRandomForest(Tree)(Tree tree, size_t nTree) {
    return RandomForest!(Tree)(tree, nTree);
}

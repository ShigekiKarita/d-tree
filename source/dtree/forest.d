module dtree.forest;

// import dtree.tree : ClassificationTree;
import mir.ndslice : iota, ndarray;
import numir : permutation, zeros;
import dtree.impurity : entropy;

struct RandomForest(Tree) {
    Tree initTree;
    size_t nTree = 5;
    Tree[] trees;

    void fit(alias ImpurityFun=entropy, Xs, Ys)(Xs xs, Ys ys) in {
        assert(xs.length == ys.length);
    } body {
        import std.math : ceil;
        import std.algorithm : min;
        import std.conv : to;
        auto ps = ys.length.permutation;
        const stride = ceil(ys.length.to!double / this.nTree).to!size_t;
        for (size_t t = 0; t < this.nTree; ++t) {
            auto tree = initTree;
            auto a = t * stride;
            auto b = min(a + stride, ps.length);
            auto ab = ps[a .. b];
            tree.fit!ImpurityFun(xs[ab], ys[ab]);
            this.trees ~= [tree];
        }
    }

    auto predict(X)(X x) {
        import mir.ndslice : map, iota, sliced;
        import mir.ndslice.allocation : ndarray;
        import mir.math : sum;
        auto result = zeros(this.initTree.nClass);
        foreach (t; this.trees) {
            result[] += t.predict(x).sliced;
        }
        result[] /= result.sum!"fast";
        return result.ndarray;
    }
}

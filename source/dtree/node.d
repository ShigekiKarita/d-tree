module dtree.node;

import dtree.decision : DecisionInfo;
import mir.random : rand;

struct Node {
    size_t depth = 0;
    DecisionInfo info;
    alias info this;

    size_t bestFeatId = 0;
    size_t bestSampleId = 0;
    double bestThreshold = 0;

    typeof(this)* left, right;

    @property
    const nElement() pure {
        return index.length;
    }

    @property
    const isLeaf() pure {
        return left is null && right is null;
    }

    auto born(DecisionInfo info) {
        return new typeof(this)(this.depth + 1, info);
    }

    auto predict(X)(X x) pure {
        if (this.isLeaf) { return this.prediction; }
        auto next = x[this.bestFeatId] > this.bestThreshold ? right : left;
        return next.predict(x);
    }

    void fit(Decision, Xs, Ys)(Xs xs, Ys ys) in {
        assert(this.index.length > 0);
        assert(xs.length == ys.length);
    } out {
        import std.array : array;
        import std.algorithm : sort;
        assert(sort(this.left.index ~ this.right.index).array == this.index);
    } do {
        import std.math : isNaN;

        auto bestImpurity = double.nan;
        Decision bestDecision;
        // TODO support discrete feat
        for (size_t fid = 0; fid < xs[0].length; ++fid) {
            foreach (sid; this.index) {
                auto x = xs[sid];
                Decision decision;
                decision.fit(x, xs, ys, this.index, fid, this.prediction.length);
                auto imp = decision.left.impurity + decision.right.impurity;
                auto equalUpdate = imp == bestImpurity && rand!bool();
                if (bestImpurity.isNaN || imp < bestImpurity || equalUpdate) {
                    bestDecision = decision;
                    bestImpurity = imp;
                    this.bestSampleId = sid;
                    this.bestFeatId = fid;
                    this.bestThreshold = decision.threshold;
                }
            }
        }

        with (bestDecision) {
            import std.stdio : writefln;
            writefln("depth: %d, impurity: %f, threshold %f, predict: %s, L: %d, R: %d, All: %d",
                     this.depth, this.impurity, this.bestThreshold, this.prediction,
                     left.index.length, right.index.length, this.index.length);
            this.left = this.born(left);
            this.right = this.born(right);
        }
    }
}

module dtree.node;


struct Node {
    size_t[] index;
    size_t depth = 0;
    double[] prediction;

    size_t bestFeatId = 0;
    size_t bestSampleId = 0;
    double bestThreshold = 0;
    double bestImpurity = double.nan;
    typeof(this)* left, right;

    @property
    const nElement() pure {
        return index.length;
    }

    @property
    const isLeaf() pure {
        return left is null && right is null;
    }

    auto born(size_t[] newIndex, double[] newPredict) {
        return new typeof(this)(newIndex, this.depth + 1, newPredict);
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
        this.bestImpurity = double.nan;
        Decision bestDecision;

        foreach (sid; this.index) {
            auto x = xs[sid];
            // TODO support discrete feat
            for (size_t fid = 0; fid < x.length; ++fid) {
                Decision decision;
                decision.fit(x, xs, ys, this.index, fid, this.prediction.length);
                if (this.bestImpurity.isNaN || decision.impurity < this.bestImpurity) {
                    // TODO randomly update when impurity == this.bestImpurity
                    bestDecision = decision;
                    this.bestImpurity = decision.impurity;
                    this.bestSampleId = sid;
                    this.bestFeatId = fid;
                    this.bestThreshold = decision.threshold;
                }
            }
        }

        with (bestDecision) {
            import std.stdio : writefln;
            writefln("depth: %d, impurity: %f, threshold %f, predict: %s, L: %d, R: %d, All: %d",
                     this.depth, this.bestImpurity, this.bestThreshold, this.prediction,
                     leftIndex.length, rightIndex.length, this.index.length);
            this.left = this.born(leftIndex, leftPrediction);
            this.right = this.born(rightIndex, rightPrediction);
        }
    }
}

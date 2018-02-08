module dtree.node;

import std.stdio : writefln;

import mir.ndslice.allocation : ndarray;
import numir : ones, zeros, Ndim;


mixin template NodeMixin() {
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
}


struct RegressionNode {
    mixin NodeMixin;

    void fit(alias ImpurityFun, Xs, Ys)(Xs xs, Ys ys) in {
        assert(this.index.length > 0);
        assert(xs.length == ys.length);
    } out {
        import std.array : array;
        import std.algorithm : sort;
        // assert(sort(this.left.index ~ this.right.index).array == this.index);
    } do {

    }
}

auto uniformProb(size_t nClass) pure {
    return ndarray(ones!double(nClass) / nClass);
}

auto normalizeProb(T)(T probs) pure {
    import mir.math : sum;
    auto psum = probs.sum!"fast";
    return psum > 0.0 ? probs / psum : probs / 1.0;
}


struct ClassificationNode {
    mixin NodeMixin;

    void fit(alias ImpurityFun, Xs, Ys)(Xs xs, Ys ys, size_t nClass) in {
        assert(this.index.length > 0);
        assert(xs.length == ys.length);
    } out {
        import std.array : array;
        import std.algorithm : sort;
        assert(sort(this.left.index ~ this.right.index).array == this.index);
    } do {
        import std.math : isNaN;
        size_t[] lbestIndex, rbestIndex;
        auto lbestProbs = uniformProb(nClass);
        auto rbestProbs = uniformProb(nClass);
        this.bestImpurity = double.nan;

        foreach (sid; this.index) {
            auto x = xs[sid];
            // TODO support discrete feat
            for (size_t fid = 0; fid < x.length; ++fid) {
                auto threshold = x[fid];
                auto lprobs = zeros!double(nClass);
                auto rprobs = zeros!double(nClass);
                size_t[] lindex, rindex;
                foreach (i; this.index) {
                    if (xs[i][fid] > threshold) {
                        rindex ~= [i];
                        ++rprobs[ys[i]];
                    } else {
                        lindex ~= [i];
                        ++lprobs[ys[i]];
                    }
                }
                lprobs[] = lprobs.normalizeProb;
                rprobs[] = rprobs.normalizeProb;
                const impurity = (ImpurityFun(lprobs) * lindex.length +
                                  ImpurityFun(rprobs) * rindex.length) / xs.length;
                if (this.bestImpurity.isNaN || impurity < this.bestImpurity) {
                    // TODO randomly update when impurity == this.bestImpurity
                    this.bestImpurity = impurity;
                    this.bestSampleId = sid;
                    this.bestFeatId = fid;
                    this.bestThreshold = threshold;
                    lbestIndex = lindex.dup;
                    rbestIndex = rindex.dup;
                    lbestProbs = lprobs.ndarray;
                    rbestProbs = rprobs.ndarray;
                }
            }
        }
        writefln("depth: %d, impurity: %f, probs: %s, L: %d, R: %d, All: %d",
                 this.depth, this.bestImpurity, this.prediction,
                 lbestIndex.length, rbestIndex.length, this.index.length);
        this.left = this.born(lbestIndex, lbestProbs);
        this.right = this.born(rbestIndex, rbestProbs);
    }
}

module dtree.tree;

import std.algorithm.comparison : min;
import std.stdio : writeln, writefln;

import mir.math : sum;
import mir.ndslice : ipack, unpack, ndmap = map;
import mir.ndslice.slice : sliced;
import mir.ndslice.allocation : ndarray;
import numir : ones, zeros, Ndim;

import dtree.impurity : gini, entropy;

struct RegressionTree {
    size_t maxDepth = 5;
    size_t minElement = 0;

    auto fit(Xs, Ys)(Xs xs, Ys ys) {
        
    }

    auto predict(X)(X x) {
        
    }
}

auto uniformProb(T=double)(size_t nClass) {
    return ndarray(ones!T(nClass) / nClass);
}


struct ClassificationNode {
    size_t[] index;
    size_t depth = 0;
    double[] probs;

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

    auto born(size_t[] newIndex, double[] newProbs) {
        return new typeof(this)(newIndex, this.depth + 1, newProbs);
    }

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
                auto lsum = lprobs.sum!"fast";
                if (lsum > 0.0) {
                    lprobs[] /= lsum;
                }
                auto rsum = rprobs.sum!"fast";
                if (rsum > 0.0) {
                    rprobs[] /= rsum;
                }
                const impurity = (ImpurityFun(lprobs) * lindex.length +
                                  ImpurityFun(rprobs) * rindex.length) / xs.length;
                // writeln("left: ", lprobs, "right: ", rprobs);
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
                 this.depth, this.bestImpurity, this.probs,
                 lbestIndex.length, rbestIndex.length, this.index.length);
        this.left = this.born(lbestIndex, lbestProbs);
        this.right = this.born(rbestIndex, rbestProbs);
    }

    auto predict(X)(X x) {
        if (this.isLeaf) { return this; }
        auto next = x[this.bestFeatId] > this.bestThreshold ? right : left;
        return next.predict(x);
    }
}

struct ClassificationTree {
    size_t nClass = 2;
    size_t maxDepth = 5;
    size_t minElement = 0;
    alias NodeT = ClassificationNode;
    NodeT* root;

    auto fit(alias ImpurityFun=entropy, Xs, Ys)(Xs xs, Ys ys) {
        void fitrec(NodeT* node) {
            if (node.depth >= this.maxDepth || node.nElement <= this.minElement) return;
            node.fit!ImpurityFun(xs, ys, this.nClass);
            fitrec(node.left);
            fitrec(node.right);
        }

        import mir.ndslice : iota;
        auto points = iota(ys.length).ndarray;
        this.root = new NodeT(points, 0, this.nClass.uniformProb);
        fitrec(this.root);
    }

    auto predict(X)(X x) {
        return this.root.predict(x).probs;
    }
}

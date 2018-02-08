module dtree.tree;

import dtree.impurity : gini, entropy;
import dtree.node;

struct RegressionTree {
    size_t maxDepth = 5;
    size_t minElement = 0;

    auto fit(Xs, Ys)(Xs xs, Ys ys) {
        
    }

    auto predict(X)(X x) pure {
        
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

        import std.range : iota;
        import std.array : array;
        auto points = iota(ys.length).array;
        this.root = new NodeT(points, 0, this.nClass.uniformProb);
        fitrec(this.root);
    }

    auto predict(X)(X x) pure {
        return this.root.predict(x);
    }
}

module dtree.tree;

import dtree.impurity : entropy;
import dtree.traits : isDecisionPolicy;
import dtree.node : Node;
import dtree.decision : Regression, Classification;

struct DecisionTree(DecisionPolicy) {
    static assert(isDecisionPolicy!DecisionPolicy);
    size_t nOutput = 2;
    size_t maxDepth = 5;
    size_t minElement = 0;
    Node* root;

    auto fit(Xs, Ys)(Xs xs, Ys ys) {
        void fitrec(Node* node) {
            if (node.depth >= this.maxDepth || node.nElement <= this.minElement) return;
            node.fit!DecisionPolicy(xs, ys);
            fitrec(node.left);
            fitrec(node.right);
        }

        import std.range : iota;
        import std.array : array;
        import numir : ones;
        import mir.ndslice : ndarray;

        auto points = iota(ys.length).array;
        auto initProb = ones(this.nOutput) * double.nan;
        this.root = new Node(points, 0, initProb.ndarray);
        fitrec(this.root);
    }

    auto predict(X)(X x) pure {
        return this.root.predict(x);
    }
}

alias ClassificationTree(alias ImpurityFun=entropy) = DecisionTree!(Classification!ImpurityFun);
alias RegressionTree = DecisionTree!Regression;

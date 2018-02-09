module dtree.traits;

import dtree.decision : DecisionInfo;

enum hasDecisionInfo(P) = is(typeof(P.init.left) == DecisionInfo) && is(typeof(P.init.right) == DecisionInfo);

enum isRegressionPolicy(P) = hasDecisionInfo!P && is(typeof({
            import numir : zeros;
            auto xs = zeros(3, 3);
            auto ys = zeros(3, 1);
            size_t[] id = [0, 2];
            P p;
            p.fit(xs[0], xs, ys, id, 0, 1);
        }));

enum isClassificationPolicy(P) = hasDecisionInfo!P && is(typeof({
            import numir : zeros;
            auto xs = zeros(3, 3);
            auto ys = zeros!int(3);
            size_t[] id = [0, 2];
            P p;
            p.fit(xs[0], xs, ys, id, 0, 1);
        }));

enum isDecisionPolicy(P) = isRegressionPolicy!P || isClassificationPolicy!P;

@safe @nogc
unittest {
    import dtree.decision;
    import dtree.impurity;

    static assert(isRegressionPolicy!Regression);
    static assert(isClassificationPolicy!(Classification!entropy));
    static assert(isClassificationPolicy!(Classification!gini));
}

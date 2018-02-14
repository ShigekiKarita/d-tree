module dtree.boosting;

// TODO add this to numir
@nogc auto mean(Result=double, Xs)(Xs xs) pure {
    import mir.math.sum : sum;
    import mir.ndslice.topology : as;
    import numir : size;
    return xs.as!Result.sum!"fast" / xs.size;
}

auto mean(ptrdiff_t axis, Result=double, Xs)(Xs xs) pure {
    import mir.ndslice : ipack, swapped, shape, each;
    import numir : size, zeros;
    auto xt = xs.swapped!(0, axis);
    auto ret = zeros!Result(xt[0].shape);
    xt.ipack!1.each!((x) {
        ret[] += x;
    });
    ret[] /= xs.length!axis;
    return ret;
}

pure @safe
unittest {
    import mir.ndslice : as, iota;
    /*
      [[0,1,2],
       [3,4,5]]
     */
    assert(iota(2, 3).mean == (5.0 / 2.0));
    assert(iota(2, 3).mean!0 == [(0.0+3.0)/2.0, (1.0+4.0)/2.0, (2.0+5.0)/2.0]);
    assert(iota(2, 3).mean!1 == [(0.0+1.0+2.0)/3.0, (3.0+4.0+5.0)/3.0]);
}


auto mseGrad(T, P)(T target, P pred) {
    return target - pred;
}

struct GradientBoosting(Model, alias gradient=mseGrad) {
    Model bluePrint;
    size_t nBoost;
    Model[] models;
    double[] initPred;
    double stepSize = -1e-2;

    auto fit(Xs, Ys)(Xs xs, Ys ys) {
        import numir : zeros_like;
        import std.range : enumerate;
        import mir.ndslice : slice, map, ipack, sliced, each;
        import mir.ndslice.topology : repeat;
        auto pred = zeros_like(ys);
        auto grad = zeros_like(ys);
        // TODO support multi dim
        this.initPred = [ys.mean];
        pred[] = this.initPred[0];
        grad[] = -gradient(ys, pred);
        models.length = nBoost;
        foreach (ref m; models) {
            m = bluePrint; // copy
            m.fit(xs, grad);
            // TODO implement line-search to find the best stepSize
            auto nextPred = zeros_like(ys);
            foreach(i, x; xs.enumerate) {
                pred[i][] += this.stepSize * m.predict(x).sliced;
            }
            grad[] = -gradient(ys, pred);
        }
    }

    auto predict(X)(X x) {
        import mir.ndslice : sliced;
        auto pred = this.initPred.sliced;
        foreach (m; models) {
            pred[] += this.stepSize * m.predict(x).sliced;
        }
        // TODO support multi dim
        return pred;
    }
}

import std.stdio;


void plot(Model, Xs, Ys)(string path, Model model, Xs xs, Ys ys, size_t resolution=100) {
    import ggplotd.aes : aes;
    import ggplotd.axes : xaxisLabel, yaxisLabel;
    import ggplotd.ggplotd : GGPlotD, putIn;
    import ggplotd.geom : geomPoint, geomLine;
    import ggplotd.colour : colourGradient;
    import ggplotd.colourspace : XYZ;

    import std.algorithm : map, cartesianProduct, minElement, maxElement;
    import std.array : array;
    import std.range : iota;

    import mir.math : sin;
    import mir.ndslice : ndarray, ndmap = map, pack, slice, sliced;
    import numir : empty, unsqueeze, arange, linspace;

    const xmin = minElement(xs[0..$, 0]);
    const xmax = maxElement(xs[0..$, 0]);
    const xstep = (xmax - xmin) / resolution;
    auto grid = linspace!double(xmin, xmax, resolution); // .slice.unsqueeze!1;
    // auto gridPreds = grid.ipack!1.ndmap!(i => model.predict(i)).ndarray;

    GGPlotD gg;

    // plot regression lines
    // gg = iota(grid.length)
    //     .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], gridPreds[i][0], 1.0, 1.0))
    //     .geomPoint
    //     .putIn(gg);

    // plot ground truth (sin function)
    gg = iota(xs.shape[0])
        .map!(i => aes!("x", "y", "colour", "size")(grid[i], sin(grid[i]), 1.0, 1))
        .geomLine
        .putIn(gg);

    // plot data point
    gg = iota(xs.shape[0])
        .map!(i => aes!("x", "y", "colour", "size")(xs[i,0], ys[i,0], 0.5, 0.1))
        .geomPoint
        .putIn(gg);

    // set colour scheme
    gg = colourGradient!XYZ( "cornflowerBlue-crimson" )
        .putIn(gg);

    gg.save(path);
    writeln("saved to " ~ path);
}

void main() {
    import mir.ndslice : map, slice, iota;
    import mir.random : Random, unpredictableSeed;
    import mir.random.variable : BernoulliVariable ;
    import mir.math : sin;
    import numir.random : normal;

    import dtree.tree : RegressionTree;

    auto nsamples = 100;
    auto ndim = 1;
    auto xs = normal(nsamples, ndim).slice * 3;
    auto ys = xs.map!sin;
    auto tree = RegressionTree();
    tree.fit(xs, ys);
    plot("plot.png", tree, xs, ys);
}


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
    import mir.ndslice : ndarray, ndmap = map, pack, slice, sliced, ipack;
    import numir : empty, unsqueeze, arange, linspace;

    const xmin = minElement(xs[0..$, 0]);
    const xmax = maxElement(xs[0..$, 0]);
    const xstep = (xmax - xmin) / resolution;
    auto grid = linspace!double(xmin, xmax, resolution).slice.unsqueeze!1;
    auto gridPreds = grid.ipack!1.ndmap!(x => model.predict(x)[0]).ndarray;

    GGPlotD gg;
    // plot data point
    gg = iota(xs.shape[0])
      .map!(i => aes!("x", "y", "colour", "size")(xs[i,0], ys[i,0], 0.5, 0.01))
      .geomPoint
      .putIn(gg);

    // plot regression lines
    gg = iota(grid.length)
      .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], gridPreds[i], 0, 1))
      .geomLine
      .putIn(gg);

    // plot ground truth (sin function)
    gg = iota(grid.length)
      .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], sin(grid[i][0]), 1, 0.2))
      .geomLine
      .putIn(gg);

    // set colour scheme
    gg = colourGradient!XYZ( "cornflowerBlue-crimson" )
      .putIn(gg);

    gg.save(path);
    writeln("saved to " ~ path);
}

void main() {
    import mir.ndslice : map, slice;
    import mir.math : sin;
    import numir : normal, uniform;

    import dtree.tree : RegressionTree;
    import dtree.forest : RandomForest;

    auto nsamples = 100;
    auto ndim = 1;
    auto xs = uniform(nsamples, ndim).slice * 10;
    auto ys = xs.map!sin + normal(xs.shape) * 0.1;
    auto tree = RegressionTree(1, 10);
    tree.fit(xs, ys);
    // plot("plot_tree.png", tree, xs, ys);

    // auto forest = RandomForest!RegressionTree(tree, 3);
    // forest.fit(xs, ys);
    // plot("plot_forest.png", forest, xs, ys);
}


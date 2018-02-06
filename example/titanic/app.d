import std.stdio;

auto readCSV(in string path) {
    import std.csv;
    import std.algorithm;
    import std.array;
    auto file = File("../data/titanic/train.csv", "r");
    auto dataset = file.byLine.joiner("\n").csvReader!(string[string])(null);
    return dataset;
}

auto attr(CSV)(CSV csv, in string key) pure {
    import std.algorithm : map;
    return csv.map!(a => a[key]);
}

struct TitanicEntry {
    int passengerId;
    bool hasCabin;
    int familySize;
    bool isAlone;
    string embarked;
    double fare;
    // string[string] raw;

    /**
       Diego Milla
       Introduction to Decision Trees (Titanic dataset)
       https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset/
     */
    this(string[string] raw, double fareMedian=0.0) {
        import std.conv : to;
        this.passengerId = raw["PassengerId"].to!int;
        this.hasCabin = raw["Cabin"] != "";
        this.familySize = raw["SibSp"].to!int + raw["Parch"].to!int + 1;
        this.isAlone = this.familySize == 0;
        this.embarked = raw["Embarked"] == "" ? "S" : raw["Embarked"];
        this.fare = raw["Fare"] == "" ? fareMedian : raw["Fare"].to!double;
        // this.raw = raw;
    }
}

void main() {
    import std.array;
    import std.algorithm;
    auto trainraw = readCSV("data/titanic/train.csv").array;
    auto testraw = readCSV("data/titanic/test.csv").array;
    // auto passengerId = test.attr("PassengerId").map!(to!int).uniq.array;
    // preprocess
    auto trainset = trainraw.map!TitanicEntry.array;
    auto testset = testraw.map!TitanicEntry.array;
    testset.writeln;
}

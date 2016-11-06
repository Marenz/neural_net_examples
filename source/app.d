import std.stdio;
import mir.ndslice;
import mir.glas.common;

GlasContext glas;



void dot ( Mat, Vec, Result ) ( Mat mat, Vec vec, ref Result result )
{
    import mir.glas.l2;

    gemv!(double, double, double)(&glas, 1.0L, mat, vec.transposed[0], 0.0,
            result.transposed[0]);
}


auto sigmoid ( T ) ( T x )
{
    import std.math : exp;

    return 1.0 / (1.0 + exp(-x));
}


auto sigmoid_derived ( T ) ( T x )
{
    return x * (1.0 - x);
}

// Create a two dimensional array filled with random values
auto randomSlice ( Lengths... ) ( double min, double max, Lengths lengths )
{
    import std.random;

    auto matrix = slice!double(lengths);

    matrix.ndEach!((ref a) => a = uniform(min, max));

    return matrix;
}

alias Matrix2d = Slice!(2, double*);
alias Vector = Slice!(1, double*);



void main()
{
    auto x = slice!double([4, 3]);

    x[0][] = [0.0, 0.0 ,1.0];
    x[1][] = [0.0, 1.0 ,1.0];
    x[2][] = [1.0, 0.0 ,1.0];
    x[3][] = [1.0, 1.0 ,1.0];

    import mir.ndslice.iteration;

    auto y = slice!double([4, 1]);
    y.transposed[] = [0.0, 0.0, 0.0, 1.0];

    auto syn0 = randomSlice(-1.0, 1.0, 3, 1);

	writefln("my x: %s\nmy y: %s\nsyn0: %s", x, y, syn0);

    auto l1_error = slice!double(y.shape);
    auto l1_delta = slice!double(l1_error.shape);

    auto dot_result = slice!double(syn0.shape);

    auto l1 = slice!double([4, 1]);

    foreach (iter; 0..100000)
    {
        auto l0 = x;

        l0.dot(syn0, l1);
        l1.ndEach!((ref a)=>a=sigmoid(a));

        l1_error[] = y;
        l1_error[] -= l1;

        l1_delta[] = l1_error;

        auto zip = assumeSameStructure!("l1_delta", "l1")(l1_delta, l1);

        zip.ndEach!((z) { //writefln("l1_delta %s, l1: %s", z.l1_delta, z.l1);
                z.l1_delta = z.l1_delta * sigmoid_derived(z.l1); });

        l0.transposed.dot(l1_delta, dot_result);

        syn0[] += dot_result;
    }


    writefln("RESULT: %s", l1);
}

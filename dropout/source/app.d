import std.stdio;
import mir.ndslice;
import mir.glas.common;

GlasContext glas;



void dot ( Mat, Mat1, Result ) ( Mat mat, Mat1 mat2, ref Result result )
{
    import mir.glas.l3;

    gemm!(double, double, double)(&glas, 1.0L, mat, mat2, 0.0, result);
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
    enum HiddenNeurons = 32;

    auto x = slice!double([4, 3]);

    x[0][] = [0.0, 0.0 ,1.0];
    x[1][] = [0.0, 1.0 ,1.0];
    x[2][] = [1.0, 0.0 ,1.0];
    x[3][] = [1.0, 1.0 ,0.0];

    import mir.ndslice.iteration;

    auto y = slice!double([4, 1]);
    y[] = [[0.0], [1.0], [1.0], [0.0]];


    auto syn0 = randomSlice(-1.0, 1.0, 3, HiddenNeurons);
    auto syn1 = randomSlice(-1.0, 1.0, HiddenNeurons, 1);

	writefln("Trains XOR\nmy x: %s\nmy y: %s\nsyn0: %s\nsyn1: %s", x, y, syn0, syn1);


    auto dot_result0 = slice!double(syn0.shape);
    auto dot_result1 = slice!double(syn1.shape);

    auto l1 = slice!double([4, HiddenNeurons]);
    auto l2 = slice!double([4, 1]);

    auto l1_error = slice!double(l1.shape);
    auto l1_delta = slice!double(l1.shape);

    auto l2_error = slice!double(l2.shape);
    auto l2_delta = slice!double(l2.shape);

    writefln("######");
    foreach (iter; 0..10000)
    {
        auto l0 = x;

        l0.dot(syn0, l1);
        l1.ndEach!((ref a)=>a=sigmoid(a));

        l1.dot(syn1, l2);
        l2.ndEach!((ref a)=>a=sigmoid(a));

        l2_error[] = y;
        l2_error[] -= l2;

        // l2_delta = l2_error * sigmoid'(l2)
        {
            l2_delta[] = l2_error;
            auto zip = assumeSameStructure!("l2_delta", "l2")(l2_delta, l2);
            zip.ndEach!((z) { z.l2_delta = z.l2_delta * sigmoid_derived(z.l2); });
        }

        // l1_error = l2_delta * syn1
        l2_delta.dot(syn1.transposed, l1_error);

        // l1_delta = l1_error * sigmoid'(l1)
        {
            l1_delta[] = l1_error;
            auto zip = assumeSameStructure!("l1_delta", "l1")(l1_delta, l1);
            zip.ndEach!((z) { z.l1_delta = z.l1_delta * sigmoid_derived(z.l1); });
        }

        // sync0 += l0 * l1_delta
        {
            l0.transposed.dot(l1_delta, dot_result0);
            syn0[] += dot_result0;
        }

        // sync1 += l1 * l2_delta
        {
            l1.transposed.dot(l2_delta, dot_result1);
            syn1[] += dot_result1;
        }
    }


    writefln("RESULT: %s", l2);
}

import std.stdio;
import mir.ndslice;
import mir.ndslice.dynamic : transposed;
import mir.ndslice.topology : map;

auto dot ( Mat, Mat1, Result ) ( Mat mat, Mat1 mat2, Result result )
{
    import mir.glas.ndslice;

    gemm!(double, double, double)(1.0L, mat, mat2, 0.0, result);

    return result;
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

    import std.random;

Random gen;

// Create a two dimensional array filled with random values
auto randomSlice ( Lengths... ) ( double min, double max, Lengths lengths )
{
    auto matrix = slice!double(lengths);

    matrix.each!((ref a) { a = uniform(min, max, gen); uniform(min,
                max, gen); });

    return matrix;
}


import mir.ndslice.slice;

size_t getLength ( Lengths... ) ()
{
    size_t len;

    foreach (Len; Lengths)
        len += len;

    return len;
}



void main()
{
    import std.math;
    import util;

    writefln("Default seed: %s", gen.defaultSeed);
    gen.seed(5);

    enum BinaryDim = 3;
    enum HiddenDim = 16; //2 * BinaryDim;
    enum Alpha = 0.1;
    enum InputDim = 2;
    enum OutputDim = 1;
    enum LargestNumber = pow(2, BinaryDim);

    auto syn0 = randomSlice(-1.0, 1.0, InputDim, HiddenDim);
    auto syn1 = randomSlice(-1.0, 1.0, HiddenDim, OutputDim);
    auto synh = randomSlice(-1.0, 1.0, HiddenDim, HiddenDim);

    writefln("syn0: %.9s", syn0);
    writefln("syn1: %.9s", syn1);
    writefln("synh: %.9s", synh);
    //auto abc = dot1(syn0, syn1);

    auto syn0_update = slice!double(syn0.shape);
    syn0_update[] = 0.0;
    auto syn1_update = slice!double(syn1.shape);
    syn1_update[] = 0.0;
    auto synh_update = slice!double(synh.shape);
    synh_update[] = 0.0;

	//writefln("Trains BinADD\nmy x: %s\nmy y: %s\nsyn0: %s\nsyn1: %s", x, y, syn0, syn1);

    import std.random;
    import std.range;
    import std.algorithm;

    auto toBinRange ( ubyte val )
    {
        import std.range;
        return sequence!((a,n)=>a[0]>>n&1)(val).take(BinaryDim).retro;
    }

    //static assert(toBinRange(1).equal([0, 0, 1]));
    /*static assert(toBinRange(1).equal([0, 0, 0, 0, 0, 0, 0, 1]));
    static assert(toBinRange(2).equal([0, 0, 0, 0, 0, 0, 1, 0]));
    static assert(toBinRange(3).equal([0, 0, 0, 0, 0, 0, 1, 1]));
    static assert(toBinRange(255).equal([1, 1, 1, 1, 1, 1, 1, 1]));
    static assert(toBinRange(254).equal([1, 1, 1, 1, 1, 1, 1, 0]));*/

    auto layer_0 = slice!double(1, InputDim);
    auto layer_1 = slice!double(1, HiddenDim);
    auto layer_2 = slice!double(1, OutputDim);

    auto y = slice!double(1, OutputDim);

    auto layer_2_error = slice!double(layer_2.shape);
    auto layer_1_error = slice!double(layer_1.shape);

    auto layer_2_deltas = slice!double(BinaryDim, 1, OutputDim);
    auto layer_1_values = slice!double(BinaryDim+1, 1, HiddenDim);

    auto layer_1_delta = slice!double(layer_1.shape);

    foreach (iter; 0..100_000) //0_000)
    {
        layer_1_values[] = 0;
        writefln("ITERATION: %s\n", iter);
        double overall_error = 0;

        ubyte a_int = cast(ubyte) uniform(0, LargestNumber/2, gen);
        ubyte b_int = cast(ubyte) uniform(0, LargestNumber/2, gen);
        ubyte c_int = cast(ubyte)(a_int + b_int);

        auto a_bin = toBinRange(a_int);
        auto b_bin = toBinRange(b_int);
        auto c_bin = toBinRange(c_int);

        ubyte[BinaryDim] d_bin;

        //if (iter % 1000 == 0)
        /*{
        writefln("a: %.9s, abin: %s", a_int, a_bin);
        writefln("b: %s, bbin: %s", b_int, b_bin);
        writefln("c: %s, cbin: %s", c_int, c_bin);
        }*/

        ResizingRegionAllocator alloc;

        foreach (i, a, b, c;
            zip(iota(BinaryDim), a_bin, b_bin, c_bin).retro)
        {
            scope(exit)
                alloc.freeAll();

            layer_0[0][] = [b, a].sliced;
            layer_1[] = 0;
            layer_2[] = 0;
            y[0][] = [c].sliced;

            //writefln("layer_0: %.9s", layer_0);
            //writefln("syn0: %.9s", syn0);
            //writefln("hidden: %.9s", layer_1_values[i+1]);
            //writefln("l1vs: %s", layer_1_values);
            //writefln("l1vs[-1]: %s", i+1);

            layer_1[] = (layer_0.mtimes(syn0, alloc) +
                    layer_1_values[i+1].mtimes(synh, alloc))
                .map!sigmoid;

            layer_2[] = layer_1.mtimes(syn1, alloc).map!sigmoid;


            layer_2_error[] = y - layer_2;
            layer_2_deltas[i][] =
                layer_2_error.mtimes(layer_2.map!sigmoid_derived, alloc);
            overall_error += abs(layer_2_error[0][0]);

            d_bin[i] = cast(ubyte)round(layer_2[0][0]);
            writefln("layer2 end: %s, %s", layer_2[0], i);
            //writefln("layer2 delt: %s, %s", layer_2_deltas[i], i);

            // Store hidden layer
            layer_1_values[i][] = layer_1;
            //writefln("l1 val: %s %s", i, layer_1_values[i]);
        }

        auto d = reduce!((a,b)=>a<<1|b)(d_bin[]);

        //if (iter % 1000 == 0)
        //    writefln("result: %s .. %s", d, d_bin);

        auto future_layer_1_delta = slice!double(layer_1.shape);
        future_layer_1_delta[] = 0;


        foreach (a, b, l1_val, l1_prev_val, l2_delta; zip(a_bin,
                                                  b_bin,
                                                  layer_1_values,
                                                  layer_1_values.drop(1),
                                                  layer_2_deltas))
        {
            scope(exit)
                alloc.freeAll();

            //writefln("ba l1 val: %s ", l1_val);
            //writefln("ba prl1 val: %s ", l1_prev_val);
            writefln("[%s %s]", a, b);
            //writefln("Layer2delt: %s", l2_delta);
            //writefln("bp inp: %.9s %.9s", a, b);
            //writefln("layer1: %.9s", l1);
            //writefln("prev_layer1: %.9s", l1_prev_val);

            layer_1_delta[] = future_layer_1_delta.mtimes(synh.transposed, alloc)
                 + l2_delta.mtimes(syn1.transposed, alloc) * l1_val.map!sigmoid_derived;

            syn1_update[] += l1_val.transposed.mtimes(l2_delta, alloc);

            synh_update[] += l1_prev_val.transposed.mtimes(layer_1_delta, alloc);

            layer_0[0][] = [a, b].sliced;
            writefln("layer 0: %s", layer_0);

            syn0_update[] += layer_0.transposed.mtimes(layer_1_delta, alloc);

            future_layer_1_delta[] = layer_1_delta;

            //writefln("syn0_up: %.9s", syn0_update);
            //writefln("syn1_up: %.9s", syn1_update);
            //writefln("synh_up: %.9s", synh_update);
        }

        syn0[] += syn0_update * Alpha;
        syn1[] += syn1_update * Alpha;
        synh[] += synh_update * Alpha;

        syn0_update[] = 0;
        syn1_update[] = 0;
        synh_update[] = 0;


        //if (iter % 1000 == 0)
        {
            writefln("Iteration: %s", iter);
            writefln("%s + %s", a_int, b_int);
            writefln("Error: %.9s", overall_error);
            writefln("Pred: %s .. %s", d, d_bin);
            writefln("True: %s .. %s", c_int, c_bin);
            writefln("-----------------------");
        }

        import std.math;

        if (isNaN(overall_error))
            break;
    }
}

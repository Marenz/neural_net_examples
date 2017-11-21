import std.stdio;
import mir.ndslice;
import mir.glas.common;

GlasContext glas;



void dot ( Mat, Mat1, Result ) ( Mat mat, Mat1 mat2, Result result )
{
    import mir.glas.l3;

    gemm!(float, float, float)(&glas, 1.0L, mat, mat2, 0.0, result);
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
auto randomSlice ( Lengths... ) ( float min, float max, Lengths lengths )
{
    auto matrix = slice!float(lengths);

    matrix.ndEach!((ref a) { a = uniform(min, max, gen); uniform(min,
                max, gen); });

    return matrix;
}


/*void dot ( Mat, Mat1, Result ) ( Mat mat, Mat1 mat2, Result result )
{
    import mir.glas.l3;

    gemm!(float, float, float)(&glas, 1.0L, mat, mat2, 0.0, result);
}*/

import mir.ndslice.slice;

size_t getLength ( Lengths... ) ()
{
    size_t len;

    foreach (Len; Lengths)
        len += len;

    return len;
}



float[] tmp_buffer;

auto dot1 ( Mat1, Mat2 ) ( Mat1 m1, Mat2 m2 )
{
    import mir.glas.l3;

    static assert (m1.shape.length == m2.shape.length);
    static assert (Mat1.N[$-1] == Mat2.N[0]);

    tmp_buffer.assumeSafeAppend().length = getLengths!([m1.shape[0],
                                                        m2.shape[$-1]]);

    Slice!(Universal, [m1.shape[0], m2.shape[$-1]], tmp_buffer.ptr) tmp_result;

    gemm!(float, float, float)(&glas, 1.0L, mat1, mat2, 0.0, tmp_result);

    return tmp_result;
}

void main()
{
    import std.math;

    writefln("Default seed: %s", gen.defaultSeed);
    gen.seed(5);

    enum BinaryDim = 8;
    enum HiddenDim = 16;
    enum Alpha = 0.1;
    enum InputDim = 2;
    enum OutputDim = 1;
    enum LargestNumber = pow(2, BinaryDim);

    import mir.ndslice.iteration;

    auto syn0 = randomSlice(-1.0, 1.0, InputDim, HiddenDim);
    auto syn1 = randomSlice(-1.0, 1.0, HiddenDim, OutputDim);
    auto synh = randomSlice(-1.0, 1.0, HiddenDim, HiddenDim);

    writefln("syn0: %.9s", syn0);
    writefln("syn1: %.9s", syn1);
    writefln("synh: %.9s", synh);
    //auto abc = dot1(syn0, syn1);

    auto syn0_update = slice!float(syn0.shape);
    syn0_update[] = 0.0;
    auto syn1_update = slice!float(syn1.shape);
    syn1_update[] = 0.0;
    auto synh_update = slice!float(synh.shape);
    synh_update[] = 0.0;

	//writefln("Trains BinADD\nmy x: %s\nmy y: %s\nsyn0: %s\nsyn1: %s", x, y, syn0, syn1);

    import std.random;
    import std.range;
    import std.algorithm;

    auto toBinRange ( ubyte val )
    {
        return sequence!((a,n)=>a[0]>>n&1)(val).take(BinaryDim).retro;
    }

    auto layer_0 = slice!float(1, InputDim);
    auto layer_1 = slice!float(1, HiddenDim);
    auto layer_2 = slice!float(1, OutputDim);

    auto y = slice!float(1, OutputDim);

    auto layer_2_error = slice!float(layer_2.shape);
    auto layer_1_error = slice!float(layer_1.shape);

    auto dot_result0 = slice!float(syn0.shape);
    auto dot_result1 = slice!float(syn1.shape);
    auto dot_resulth = slice!float(synh.shape);
    auto layer2_tmp = slice!float(layer_2.shape);

    auto l2_dot_s1 = slice!float(1, HiddenDim);

    auto layer_2_deltas = slice!float(BinaryDim, 1, OutputDim);
    auto layer_1_values = slice!float(BinaryDim+1, 1, HiddenDim);
    layer_1_values[] = 0;

    auto layer_1_delta = slice!float(layer_1.shape);

    foreach (iter; 0..100_00) //0_000)
    {
        writefln("ITERATION: %s\n", iter);
        float overall_error = 0;

        ubyte a_int = cast(ubyte) 3; //uniform(0, LargestNumber/2, gen);
        ubyte b_int = cast(ubyte) 5; //uniform(0, LargestNumber/2, gen);
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

        foreach (i, a, b, c; zip(iota(BinaryDim), a_bin, b_bin, c_bin).retro)
        {
            layer_0[0][] = [b, a].sliced;
            layer_1[] = 0;
            layer_2[] = 0;
            y[0][] = [c].sliced;

            writefln("layer_0: %.9s", layer_0);
            writefln("syn0: %.9s", syn0);
            writefln("hidden: %.9s", layer_1_values[i+1]);
            //writefln("y: %s", y);

            // layer_1 = sig(layer_0 * syn0 + layer_1_vals[$-1] * synh)
            {
                layer_0.dot(syn0, layer_1);
                layer_1_values[i+1].dot(synh, dot_result1);

                layer_1[] += dot_result1;
                layer_1.ndEach!((ref a)=>a=sigmoid(a));
            }

            // layer_2 = layer_1 * syn1
            {
                layer_1.dot(syn1, layer_2);
                writefln("layer_1: %.9s", layer_1);
                writefln("syn1: %.9s", syn1);
                writefln("l2: %.9s", layer_2);


                layer_2.ndEach!((ref a)=>a=sigmoid(a));
            }

            // layer_2_deltas[i] = (y - layer_2) * sigmoid'(layer_2)
            {
                layer_2_deltas[i][] = y;
                layer_2_deltas[i][] -= layer_2;

                writefln("layer_2 %.9s", layer_2);
                writefln("y       %.9s", y);
                writefln("error:  %.9s", layer_2_deltas[i]);

                import std.math : abs;
                overall_error += abs(layer_2_deltas[i][0][0]);

                auto zip = assumeSameStructure!("l2_delta",
                        "l2")(layer_2_deltas[i][], layer_2);
                zip.ndEach!((z) { z.l2_delta = z.l2_delta * sigmoid_derived(z.l2); });
                writefln("delta:  %.9s", layer_2_deltas[i]);
            }

            d_bin[i] = cast(ubyte)round(layer_2[0][0]);
            writefln("layer2 end: %.9s", layer_2[0]);

            // Store hidden layer
            layer_1_values[i][] = layer_1;
        }

        auto d = d_bin[].reduce!((a,b)=>a<<1|b);

        //if (iter % 1000 == 0)
        //    writefln("result: %s .. %s", d, d_bin);

        auto future_layer_1_delta = slice!float(layer_1.shape);
        future_layer_1_delta[] = 0;

        foreach (a, b, l1, l1_prev_val, l2_delta; zip(a_bin,
                                                  b_bin,
                                                  layer_1_values,
                                                  layer_1_values.drop(1),
                                                  layer_2_deltas))
        {
            writefln("bp inp: %.9s %.9s", a, b);
            writefln("layer1: %.9s", l1);
            writefln("prev_layer1: %.9s", l1_prev_val);

            // layer_1_delta = future_layer_1_delta.dot(syn_h.T) +
            //                 l2_delta.dot(syn_1.T) * sigmoid'(layer_1)
            {
                // 1 x HiddenDim * HiddenDim x HiddenDim => 1 x HiddenDim
                future_layer_1_delta.dot(synh.transposed, layer_1_delta);
                //writefln("future_1_delta: %s", future_layer_1_delta);
                //writefln("synh.tr: %s", synh.transposed);
                //writefln("future_l1_delta dot synh: %s", layer_1_delta);
                l2_delta.dot(syn1.transposed, l2_dot_s1);

                auto zip = assumeSameStructure!("l2_dot_s1",
                        "l1")(l2_dot_s1, l1);
                zip.ndEach!((z) { z.l2_dot_s1 = z.l2_dot_s1 * sigmoid_derived(z.l1); });

                layer_1_delta[] += l2_dot_s1;
            }

            l1.transposed.dot(l2_delta, dot_result1);
            syn1_update[] += dot_result1;

            l1_prev_val.transposed.dot(layer_1_delta, dot_resulth);
            synh_update[] += dot_resulth;

            layer_0[0][] = [a, b].sliced;
            //writefln("l1_delta: %.9s", layer_1_delta);
            layer_0.transposed.dot(layer_1_delta, dot_result0);
            syn0_update[] += dot_result0;

            future_layer_1_delta[] = layer_1_delta;

            writefln("syn0_up: %.9s", syn0_update);
            writefln("syn1_up: %.9s", syn1_update);
            writefln("synh_up: %.9s", synh_update);
        }

        syn0_update[] *= Alpha;
        syn1_update[] *= Alpha;
        synh_update[] *= Alpha;

        syn0[] += syn0_update;
        syn1[] += syn1_update;
        synh[] += synh_update;

        syn0_update[] = 0;
        syn1_update[] = 0;
        synh_update[] = 0;


        //if (iter % 1000 == 0)
        {
            writefln("Iteration: %s", iter);
            writefln("%s + %s", a_int, b_int);
            writefln("Error: %.9s", overall_error);
            writefln("Pred: %.9s .. %.9s", d, d_bin);
            writefln("True: %.9s .. %.9s", c_int, c_bin);
            writefln("-----------------------");
        }

        import std.math;
        if (isnan(overall_error))
            break;
    }

    /+auto dot_result0 = slice!float(syn0.shape);
    auto dot_result1 = slice!float(syn1.shape);

    auto l1_error = slice!float(l1.shape);
    auto l1_delta = slice!float(l1.shape);

    auto l2_error = slice!float(l2.shape);
    auto l2_delta = slice!float(l2.shape);

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


    writefln("RESULT: %.9s", l2);
+/
}

# ------------------------------------------------------------
"""
    didactic_MO01UKP()

Setup a didactic instance of MO-01UKP 
"""
function didactic_MO01UKP()

    p = [ 13 10  3 16 12 11  1  9 19 13 ;     # profit 1
           1 10  3 13 12 19 16 13 11  9  ]    # profit 2
    w  = [ 4, 4, 3, 5, 5, 3, 2, 3, 5, 4  ]    # weight
    c  = 19                                   # capacity

    return p, w, c
end 


# ------------------------------------------------------------
"""
    generate_MO01UKP(n = 10, o = 2, max_ci = 100, max_wi = 30)

Generate randomly an instance for the MO-01UKP with c and w uniformly distributed
"""
function generate_MO01UKP(n = 10, o = 2, max_ci = 100, max_wi = 30)

    p = rand(1:max_ci,o,n) # c_i \in [1,max_ci]   # profits
    w = rand(1:max_wi,n) # w_i \in [1,max_wi]     # weight
    c = round(Int64, sum(w)/2)                    # capacity
                
    return p, w, c
end


# ------------------------------------------------------------
"""
    TamVan_MO01UKP()

Setup the instance MOKP_p-6_n-30_1.dat of MO-01UKP 
"""
function TamVan_MO01UKP()

    p = [ 42  74  52  57  41  83  89  74  87  49  3  9  57  18  26  8  26  5  76  26  19  87  81  28  89  10  58  71  49  21;
          34  84  51  77  99  70  48  72  60  85  45  36  9  51  70  51  58  12  47  86  21  39  20  96  97  40  16  74  34  3;
          99  98  8  95  84  96  72  60  64  3  63  62  13  26  96  67  99  26  4  35  3  11  81  77  26  3  57  38  81  72;
          85  47  66  66  76  18  44  91  94  75  20  96  28  61  39  89  48  97  43  27  3  18  29  18  18  21  51  60  61  51;
          12  8  25  83  16  77  99  25  25  25  66  69  60  31  2  26  44  17  42  18  47  94  71  84  12  58  15  74  38  72;
          74  98  68  17  20  39  42  95  10  88  59  23  27  73  55  2  40  23  98  58  45  19  99  89  89  82  69  84  49  62]
    w  = [9, 45, 74, 75, 86, 5, 62, 89, 84, 5, 75, 41, 77, 68, 7, 75, 84, 97, 86, 79, 48, 44, 15, 22, 11, 44, 77, 22, 34, 27]    # weight
    c  = 783 #783.5                                   # capacity

    return p, w, c
end 



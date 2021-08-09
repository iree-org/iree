module  {
  flow.variable @"__iree_flow___sm_node260__m.layer-2.kernel" dense<1.000000e+00> : tensor<3x3x3x16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node266__m.layer-3.gamma" dense<5.000000e-01> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node267__m.layer-3.beta" dense<0.333333343> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node268__m.layer-3.moving_mean" dense<2.500000e-01> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node269__m.layer-3.moving_variance" dense<2.000000e-01> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node288__m.layer-9.depthwise_kernel" dense<0.166666672> : tensor<3x3x16x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node294__m.layer-10.gamma" dense<0.142857149> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node295__m.layer-10.beta" dense<1.250000e-01> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node296__m.layer-10.moving_mean" dense<0.111111112> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node297__m.layer-10.moving_variance" dense<1.000000e-01> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node314__m.layer-14.kernel" dense<0.0909090936> : tensor<1x1x16x8xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node315__m.layer-14.bias" dense<0.0833333358> : tensor<8xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node324__m.layer-16.kernel" dense<0.0769230798> : tensor<1x1x8x16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node325__m.layer-16.bias" dense<0.0714285746> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node340__m.layer-21.kernel" dense<0.0666666701> : tensor<1x1x16x16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node346__m.layer-22.gamma" dense<6.250000e-02> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node347__m.layer-22.beta" dense<0.0588235296> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node348__m.layer-22.moving_mean" dense<0.055555556> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node349__m.layer-22.moving_variance" dense<0.0526315793> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node354__m.layer-23.kernel" dense<5.000000e-02> : tensor<1x1x16x72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node360__m.layer-24.gamma" dense<0.0476190485> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node361__m.layer-24.beta" dense<0.0454545468> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node362__m.layer-24.moving_mean" dense<0.0434782617> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node363__m.layer-24.moving_variance" dense<0.0416666679> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node376__m.layer-27.depthwise_kernel" dense<4.000000e-02> : tensor<3x3x72x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node382__m.layer-28.gamma" dense<0.0384615399> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node383__m.layer-28.beta" dense<0.0370370373> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node384__m.layer-28.moving_mean" dense<0.0357142873> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node385__m.layer-28.moving_variance" dense<0.0344827585> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node394__m.layer-30.kernel" dense<0.0333333351> : tensor<1x1x72x24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node400__m.layer-31.gamma" dense<0.0322580636> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node401__m.layer-31.beta" dense<3.125000e-02> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node402__m.layer-31.moving_mean" dense<0.0303030312> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node403__m.layer-31.moving_variance" dense<0.0294117648> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node408__m.layer-32.kernel" dense<0.0285714287> : tensor<1x1x24x88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node414__m.layer-33.gamma" dense<0.027777778> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node415__m.layer-33.beta" dense<0.0270270277> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node416__m.layer-33.moving_mean" dense<0.0263157897> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node417__m.layer-33.moving_variance" dense<0.025641026> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node426__m.layer-35.depthwise_kernel" dense<2.500000e-02> : tensor<3x3x88x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node432__m.layer-36.gamma" dense<0.024390243> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node433__m.layer-36.beta" dense<0.0238095243> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node434__m.layer-36.moving_mean" dense<0.0232558139> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node435__m.layer-36.moving_variance" dense<0.0227272734> : tensor<88xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node444__m.layer-38.kernel" dense<0.0222222228> : tensor<1x1x88x24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node450__m.layer-39.gamma" dense<0.0217391308> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node451__m.layer-39.beta" dense<0.0212765951> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node452__m.layer-39.moving_mean" dense<0.020833334> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node453__m.layer-39.moving_variance" dense<0.0204081628> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node462__m.layer-41.kernel" dense<2.000000e-02> : tensor<1x1x24x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node468__m.layer-42.gamma" dense<0.0196078438> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node469__m.layer-42.beta" dense<0.0192307699> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node470__m.layer-42.moving_mean" dense<0.0188679248> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node471__m.layer-42.moving_variance" dense<0.0185185187> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node490__m.layer-48.depthwise_kernel" dense<0.0181818176> : tensor<5x5x96x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node496__m.layer-49.gamma" dense<0.0178571437> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node497__m.layer-49.beta" dense<0.0175438598> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node498__m.layer-49.moving_mean" dense<0.0172413792> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node499__m.layer-49.moving_variance" dense<0.0169491526> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node522__m.layer-56.kernel" dense<0.0166666675> : tensor<1x1x96x24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node523__m.layer-56.bias" dense<0.0163934417> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node532__m.layer-58.kernel" dense<0.0161290318> : tensor<1x1x24x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node533__m.layer-58.bias" dense<0.0158730168> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node548__m.layer-63.kernel" dense<1.562500e-02> : tensor<1x1x96x40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node554__m.layer-64.gamma" dense<0.0153846154> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node555__m.layer-64.beta" dense<0.0151515156> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node556__m.layer-64.moving_mean" dense<0.0149253728> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node557__m.layer-64.moving_variance" dense<0.0147058824> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node562__m.layer-65.kernel" dense<0.0144927539> : tensor<1x1x40x240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node568__m.layer-66.gamma" dense<0.0142857144> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node569__m.layer-66.beta" dense<0.0140845068> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node570__m.layer-66.moving_mean" dense<0.013888889> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node571__m.layer-66.moving_variance" dense<0.01369863> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node586__m.layer-71.depthwise_kernel" dense<0.0135135138> : tensor<5x5x240x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node592__m.layer-72.gamma" dense<0.0133333337> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node593__m.layer-72.beta" dense<0.0131578948> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node594__m.layer-72.moving_mean" dense<0.012987013> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node595__m.layer-72.moving_variance" dense<0.012820513> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node618__m.layer-79.kernel" dense<0.0126582282> : tensor<1x1x240x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node619__m.layer-79.bias" dense<1.250000e-02> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node628__m.layer-81.kernel" dense<0.0123456791> : tensor<1x1x64x240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node629__m.layer-81.bias" dense<0.0121951215> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node644__m.layer-86.kernel" dense<0.0120481923> : tensor<1x1x240x40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node650__m.layer-87.gamma" dense<0.0119047621> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node651__m.layer-87.beta" dense<0.0117647061> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node652__m.layer-87.moving_mean" dense<0.0116279069> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node653__m.layer-87.moving_variance" dense<0.0114942528> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node662__m.layer-89.kernel" dense<0.0113636367> : tensor<1x1x40x240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node668__m.layer-90.gamma" dense<0.0112359552> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node669__m.layer-90.beta" dense<0.0111111114> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node670__m.layer-90.moving_mean" dense<0.0109890113> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node671__m.layer-90.moving_variance" dense<0.0108695654> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node686__m.layer-95.depthwise_kernel" dense<0.0107526882> : tensor<5x5x240x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node692__m.layer-96.gamma" dense<0.0106382975> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node693__m.layer-96.beta" dense<0.0105263162> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node694__m.layer-96.moving_mean" dense<0.010416667> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node695__m.layer-96.moving_variance" dense<0.010309278> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node718__m.layer-103.kernel" dense<0.0102040814> : tensor<1x1x240x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node719__m.layer-103.bias" dense<0.0101010101> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node728__m.layer-105.kernel" dense<0.00999999977> : tensor<1x1x64x240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node729__m.layer-105.bias" dense<9.900990e-03> : tensor<240xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node744__m.layer-110.kernel" dense<0.00980392192> : tensor<1x1x240x40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node750__m.layer-111.gamma" dense<0.00970873795> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node751__m.layer-111.beta" dense<0.00961538497> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node752__m.layer-111.moving_mean" dense<9.523810e-03> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node753__m.layer-111.moving_variance" dense<0.0094339624> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node762__m.layer-113.kernel" dense<0.00934579409> : tensor<1x1x40x120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node768__m.layer-114.gamma" dense<0.00925925932> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node769__m.layer-114.beta" dense<0.00917431153> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node770__m.layer-114.moving_mean" dense<0.0090909088> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node771__m.layer-114.moving_variance" dense<0.00900900922> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node786__m.layer-119.depthwise_kernel" dense<0.00892857183> : tensor<5x5x120x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node792__m.layer-120.gamma" dense<0.00884955748> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node793__m.layer-120.beta" dense<0.00877192988> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node794__m.layer-120.moving_mean" dense<0.00869565178> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node795__m.layer-120.moving_variance" dense<8.620690e-03> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node818__m.layer-127.kernel" dense<0.00854700897> : tensor<1x1x120x32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node819__m.layer-127.bias" dense<0.00847457629> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node828__m.layer-129.kernel" dense<0.00840336177> : tensor<1x1x32x120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node829__m.layer-129.bias" dense<0.00833333377> : tensor<120xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node844__m.layer-134.kernel" dense<0.00826446246> : tensor<1x1x120x48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node850__m.layer-135.gamma" dense<0.00819672085> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node851__m.layer-135.beta" dense<0.008130081> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node852__m.layer-135.moving_mean" dense<0.00806451589> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node853__m.layer-135.moving_variance" dense<8.000000e-03> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node858__m.layer-136.kernel" dense<0.00793650839> : tensor<1x1x48x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node864__m.layer-137.gamma" dense<0.00787401571> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node865__m.layer-137.beta" dense<7.812500e-03> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node866__m.layer-137.moving_mean" dense<0.00775193795> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node867__m.layer-137.moving_variance" dense<0.0076923077> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node882__m.layer-142.depthwise_kernel" dense<0.00763358781> : tensor<5x5x144x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node888__m.layer-143.gamma" dense<0.0075757578> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node889__m.layer-143.beta" dense<0.00751879718> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node890__m.layer-143.moving_mean" dense<0.00746268639> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node891__m.layer-143.moving_variance" dense<0.00740740728> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node914__m.layer-150.kernel" dense<0.0073529412> : tensor<1x1x144x40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node915__m.layer-150.bias" dense<7.299270e-03> : tensor<40xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node924__m.layer-152.kernel" dense<0.00724637694> : tensor<1x1x40x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node925__m.layer-152.bias" dense<0.00719424477> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node940__m.layer-157.kernel" dense<0.00714285718> : tensor<1x1x144x48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node946__m.layer-158.gamma" dense<0.00709219835> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node947__m.layer-158.beta" dense<0.00704225338> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node948__m.layer-158.moving_mean" dense<0.00699300691> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node949__m.layer-158.moving_variance" dense<0.0069444445> : tensor<48xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node958__m.layer-160.kernel" dense<0.0068965517> : tensor<1x1x48x288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node964__m.layer-161.gamma" dense<0.00684931502> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node965__m.layer-161.beta" dense<0.00680272094> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node966__m.layer-161.moving_mean" dense<0.00675675692> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node967__m.layer-161.moving_variance" dense<0.00671140943> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node986__m.layer-167.depthwise_kernel" dense<0.00666666683> : tensor<5x5x288x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node992__m.layer-168.gamma" dense<0.00662251655> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node993__m.layer-168.beta" dense<0.00657894742> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node994__m.layer-168.moving_mean" dense<0.00653594779> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node995__m.layer-168.moving_variance" dense<0.00649350649> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1018__m.layer-175.kernel" dense<0.0064516128> : tensor<1x1x288x72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1019__m.layer-175.bias" dense<0.00641025649> : tensor<72xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1028__m.layer-177.kernel" dense<0.00636942684> : tensor<1x1x72x288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1029__m.layer-177.bias" dense<0.00632911408> : tensor<288xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1044__m.layer-182.kernel" dense<0.00628930796> : tensor<1x1x288x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1050__m.layer-183.gamma" dense<6.250000e-03> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1051__m.layer-183.beta" dense<0.00621118024> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1052__m.layer-183.moving_mean" dense<0.00617283955> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1053__m.layer-183.moving_variance" dense<0.00613496918> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1058__m.layer-184.kernel" dense<0.00609756075> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1064__m.layer-185.gamma" dense<0.00606060587> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1065__m.layer-185.beta" dense<0.00602409616> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1066__m.layer-185.moving_mean" dense<0.00598802418> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1067__m.layer-185.moving_variance" dense<0.00595238106> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1082__m.layer-190.depthwise_kernel" dense<5.917160e-03> : tensor<5x5x576x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1088__m.layer-191.gamma" dense<0.00588235306> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1089__m.layer-191.beta" dense<0.00584795326> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1090__m.layer-191.moving_mean" dense<0.00581395347> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1091__m.layer-191.moving_variance" dense<0.00578034669> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1114__m.layer-198.kernel" dense<0.00574712642> : tensor<1x1x576x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1115__m.layer-198.bias" dense<0.00571428565> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1124__m.layer-200.kernel" dense<0.00568181835> : tensor<1x1x144x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1125__m.layer-200.bias" dense<0.00564971752> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1140__m.layer-205.kernel" dense<0.00561797759> : tensor<1x1x576x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1146__m.layer-206.gamma" dense<0.00558659201> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1147__m.layer-206.beta" dense<0.00555555569> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1148__m.layer-206.moving_mean" dense<0.00552486209> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1149__m.layer-206.moving_variance" dense<0.00549450563> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1158__m.layer-208.kernel" dense<0.00546448072> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1164__m.layer-209.gamma" dense<0.00543478271> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1165__m.layer-209.beta" dense<0.00540540554> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1166__m.layer-209.moving_mean" dense<0.00537634408> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1167__m.layer-209.moving_variance" dense<0.00534759369> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1182__m.layer-214.depthwise_kernel" dense<0.00531914877> : tensor<5x5x576x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1188__m.layer-215.gamma" dense<0.00529100513> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1189__m.layer-215.beta" dense<0.00526315812> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1190__m.layer-215.moving_mean" dense<0.00523560215> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1191__m.layer-215.moving_variance" dense<0.00520833349> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1214__m.layer-222.kernel" dense<0.00518134702> : tensor<1x1x576x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1215__m.layer-222.bias" dense<0.00515463902> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1224__m.layer-224.kernel" dense<0.00512820529> : tensor<1x1x144x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1225__m.layer-224.bias" dense<0.00510204071> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1240__m.layer-229.kernel" dense<0.00507614203> : tensor<1x1x576x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1246__m.layer-230.gamma" dense<0.00505050505> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1247__m.layer-230.beta" dense<0.00502512557> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1248__m.layer-230.moving_mean" dense<5.000000e-03> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1249__m.layer-230.moving_variance" dense<0.00497512426> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1258__m.layer-232.kernel" dense<0.00495049497> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1264__m.layer-233.gamma" dense<0.00492610829> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1265__m.layer-233.beta" dense<0.00490196096> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1266__m.layer-233.moving_mean" dense<0.00487804879> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1267__m.layer-233.moving_variance" dense<0.00485436898> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1290__m.layer-240.kernel" dense<0.00483091781> : tensor<1x1x576x1024xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1291__m.layer-240.bias" dense<0.00480769249> : tensor<1024xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1310__m.layer-246.kernel" dense<0.00478468882> : tensor<1x1x1024x1000xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1311__m.layer-246.bias" dense<0.00476190494> : tensor<1000xf32> attributes {noinline, sym_visibility = "private"}
  func @call() {
    %arg0 = util.unfoldable_constant dense<1.5> : tensor<1x224x224x3xf32>
    %0 = flow.variable.address @"__iree_flow___sm_node260__m.layer-2.kernel" : !util.ptr<tensor<3x3x3x16xf32>>
    %1 = flow.variable.address @"__iree_flow___sm_node266__m.layer-3.gamma" : !util.ptr<tensor<16xf32>>
    %2 = flow.variable.address @"__iree_flow___sm_node267__m.layer-3.beta" : !util.ptr<tensor<16xf32>>
    %3 = flow.variable.address @"__iree_flow___sm_node268__m.layer-3.moving_mean" : !util.ptr<tensor<16xf32>>
    %4 = flow.variable.address @"__iree_flow___sm_node269__m.layer-3.moving_variance" : !util.ptr<tensor<16xf32>>
    %5 = flow.variable.address @"__iree_flow___sm_node288__m.layer-9.depthwise_kernel" : !util.ptr<tensor<3x3x16x1xf32>>
    %6 = flow.variable.address @"__iree_flow___sm_node294__m.layer-10.gamma" : !util.ptr<tensor<16xf32>>
    %7 = flow.variable.address @"__iree_flow___sm_node295__m.layer-10.beta" : !util.ptr<tensor<16xf32>>
    %8 = flow.variable.address @"__iree_flow___sm_node296__m.layer-10.moving_mean" : !util.ptr<tensor<16xf32>>
    %9 = flow.variable.address @"__iree_flow___sm_node297__m.layer-10.moving_variance" : !util.ptr<tensor<16xf32>>
    %10 = flow.variable.address @"__iree_flow___sm_node314__m.layer-14.kernel" : !util.ptr<tensor<1x1x16x8xf32>>
    %11 = flow.variable.address @"__iree_flow___sm_node315__m.layer-14.bias" : !util.ptr<tensor<8xf32>>
    %12 = flow.variable.address @"__iree_flow___sm_node324__m.layer-16.kernel" : !util.ptr<tensor<1x1x8x16xf32>>
    %13 = flow.variable.address @"__iree_flow___sm_node325__m.layer-16.bias" : !util.ptr<tensor<16xf32>>
    %14 = flow.variable.address @"__iree_flow___sm_node340__m.layer-21.kernel" : !util.ptr<tensor<1x1x16x16xf32>>
    %15 = flow.variable.address @"__iree_flow___sm_node346__m.layer-22.gamma" : !util.ptr<tensor<16xf32>>
    %16 = flow.variable.address @"__iree_flow___sm_node347__m.layer-22.beta" : !util.ptr<tensor<16xf32>>
    %17 = flow.variable.address @"__iree_flow___sm_node348__m.layer-22.moving_mean" : !util.ptr<tensor<16xf32>>
    %18 = flow.variable.address @"__iree_flow___sm_node349__m.layer-22.moving_variance" : !util.ptr<tensor<16xf32>>
    %19 = flow.variable.address @"__iree_flow___sm_node354__m.layer-23.kernel" : !util.ptr<tensor<1x1x16x72xf32>>
    %20 = flow.variable.address @"__iree_flow___sm_node360__m.layer-24.gamma" : !util.ptr<tensor<72xf32>>
    %21 = flow.variable.address @"__iree_flow___sm_node361__m.layer-24.beta" : !util.ptr<tensor<72xf32>>
    %22 = flow.variable.address @"__iree_flow___sm_node362__m.layer-24.moving_mean" : !util.ptr<tensor<72xf32>>
    %23 = flow.variable.address @"__iree_flow___sm_node363__m.layer-24.moving_variance" : !util.ptr<tensor<72xf32>>
    %24 = flow.variable.address @"__iree_flow___sm_node376__m.layer-27.depthwise_kernel" : !util.ptr<tensor<3x3x72x1xf32>>
    %25 = flow.variable.address @"__iree_flow___sm_node382__m.layer-28.gamma" : !util.ptr<tensor<72xf32>>
    %26 = flow.variable.address @"__iree_flow___sm_node383__m.layer-28.beta" : !util.ptr<tensor<72xf32>>
    %27 = flow.variable.address @"__iree_flow___sm_node384__m.layer-28.moving_mean" : !util.ptr<tensor<72xf32>>
    %28 = flow.variable.address @"__iree_flow___sm_node385__m.layer-28.moving_variance" : !util.ptr<tensor<72xf32>>
    %29 = flow.variable.address @"__iree_flow___sm_node394__m.layer-30.kernel" : !util.ptr<tensor<1x1x72x24xf32>>
    %30 = flow.variable.address @"__iree_flow___sm_node400__m.layer-31.gamma" : !util.ptr<tensor<24xf32>>
    %31 = flow.variable.address @"__iree_flow___sm_node401__m.layer-31.beta" : !util.ptr<tensor<24xf32>>
    %32 = flow.variable.address @"__iree_flow___sm_node402__m.layer-31.moving_mean" : !util.ptr<tensor<24xf32>>
    %33 = flow.variable.address @"__iree_flow___sm_node403__m.layer-31.moving_variance" : !util.ptr<tensor<24xf32>>
    %34 = flow.variable.address @"__iree_flow___sm_node408__m.layer-32.kernel" : !util.ptr<tensor<1x1x24x88xf32>>
    %35 = flow.variable.address @"__iree_flow___sm_node414__m.layer-33.gamma" : !util.ptr<tensor<88xf32>>
    %36 = flow.variable.address @"__iree_flow___sm_node415__m.layer-33.beta" : !util.ptr<tensor<88xf32>>
    %37 = flow.variable.address @"__iree_flow___sm_node416__m.layer-33.moving_mean" : !util.ptr<tensor<88xf32>>
    %38 = flow.variable.address @"__iree_flow___sm_node417__m.layer-33.moving_variance" : !util.ptr<tensor<88xf32>>
    %39 = flow.variable.address @"__iree_flow___sm_node426__m.layer-35.depthwise_kernel" : !util.ptr<tensor<3x3x88x1xf32>>
    %40 = flow.variable.address @"__iree_flow___sm_node432__m.layer-36.gamma" : !util.ptr<tensor<88xf32>>
    %41 = flow.variable.address @"__iree_flow___sm_node433__m.layer-36.beta" : !util.ptr<tensor<88xf32>>
    %42 = flow.variable.address @"__iree_flow___sm_node434__m.layer-36.moving_mean" : !util.ptr<tensor<88xf32>>
    %43 = flow.variable.address @"__iree_flow___sm_node435__m.layer-36.moving_variance" : !util.ptr<tensor<88xf32>>
    %44 = flow.variable.address @"__iree_flow___sm_node444__m.layer-38.kernel" : !util.ptr<tensor<1x1x88x24xf32>>
    %45 = flow.variable.address @"__iree_flow___sm_node450__m.layer-39.gamma" : !util.ptr<tensor<24xf32>>
    %46 = flow.variable.address @"__iree_flow___sm_node451__m.layer-39.beta" : !util.ptr<tensor<24xf32>>
    %47 = flow.variable.address @"__iree_flow___sm_node452__m.layer-39.moving_mean" : !util.ptr<tensor<24xf32>>
    %48 = flow.variable.address @"__iree_flow___sm_node453__m.layer-39.moving_variance" : !util.ptr<tensor<24xf32>>
    %49 = flow.variable.address @"__iree_flow___sm_node462__m.layer-41.kernel" : !util.ptr<tensor<1x1x24x96xf32>>
    %50 = flow.variable.address @"__iree_flow___sm_node468__m.layer-42.gamma" : !util.ptr<tensor<96xf32>>
    %51 = flow.variable.address @"__iree_flow___sm_node469__m.layer-42.beta" : !util.ptr<tensor<96xf32>>
    %52 = flow.variable.address @"__iree_flow___sm_node470__m.layer-42.moving_mean" : !util.ptr<tensor<96xf32>>
    %53 = flow.variable.address @"__iree_flow___sm_node471__m.layer-42.moving_variance" : !util.ptr<tensor<96xf32>>
    %54 = flow.variable.address @"__iree_flow___sm_node490__m.layer-48.depthwise_kernel" : !util.ptr<tensor<5x5x96x1xf32>>
    %55 = flow.variable.address @"__iree_flow___sm_node496__m.layer-49.gamma" : !util.ptr<tensor<96xf32>>
    %56 = flow.variable.address @"__iree_flow___sm_node497__m.layer-49.beta" : !util.ptr<tensor<96xf32>>
    %57 = flow.variable.address @"__iree_flow___sm_node498__m.layer-49.moving_mean" : !util.ptr<tensor<96xf32>>
    %58 = flow.variable.address @"__iree_flow___sm_node499__m.layer-49.moving_variance" : !util.ptr<tensor<96xf32>>
    %59 = flow.variable.address @"__iree_flow___sm_node522__m.layer-56.kernel" : !util.ptr<tensor<1x1x96x24xf32>>
    %60 = flow.variable.address @"__iree_flow___sm_node523__m.layer-56.bias" : !util.ptr<tensor<24xf32>>
    %61 = flow.variable.address @"__iree_flow___sm_node532__m.layer-58.kernel" : !util.ptr<tensor<1x1x24x96xf32>>
    %62 = flow.variable.address @"__iree_flow___sm_node533__m.layer-58.bias" : !util.ptr<tensor<96xf32>>
    %63 = flow.variable.address @"__iree_flow___sm_node548__m.layer-63.kernel" : !util.ptr<tensor<1x1x96x40xf32>>
    %64 = flow.variable.address @"__iree_flow___sm_node554__m.layer-64.gamma" : !util.ptr<tensor<40xf32>>
    %65 = flow.variable.address @"__iree_flow___sm_node555__m.layer-64.beta" : !util.ptr<tensor<40xf32>>
    %66 = flow.variable.address @"__iree_flow___sm_node556__m.layer-64.moving_mean" : !util.ptr<tensor<40xf32>>
    %67 = flow.variable.address @"__iree_flow___sm_node557__m.layer-64.moving_variance" : !util.ptr<tensor<40xf32>>
    %68 = flow.variable.address @"__iree_flow___sm_node562__m.layer-65.kernel" : !util.ptr<tensor<1x1x40x240xf32>>
    %69 = flow.variable.address @"__iree_flow___sm_node568__m.layer-66.gamma" : !util.ptr<tensor<240xf32>>
    %70 = flow.variable.address @"__iree_flow___sm_node569__m.layer-66.beta" : !util.ptr<tensor<240xf32>>
    %71 = flow.variable.address @"__iree_flow___sm_node570__m.layer-66.moving_mean" : !util.ptr<tensor<240xf32>>
    %72 = flow.variable.address @"__iree_flow___sm_node571__m.layer-66.moving_variance" : !util.ptr<tensor<240xf32>>
    %73 = flow.variable.address @"__iree_flow___sm_node586__m.layer-71.depthwise_kernel" : !util.ptr<tensor<5x5x240x1xf32>>
    %74 = flow.variable.address @"__iree_flow___sm_node592__m.layer-72.gamma" : !util.ptr<tensor<240xf32>>
    %75 = flow.variable.address @"__iree_flow___sm_node593__m.layer-72.beta" : !util.ptr<tensor<240xf32>>
    %76 = flow.variable.address @"__iree_flow___sm_node594__m.layer-72.moving_mean" : !util.ptr<tensor<240xf32>>
    %77 = flow.variable.address @"__iree_flow___sm_node595__m.layer-72.moving_variance" : !util.ptr<tensor<240xf32>>
    %78 = flow.variable.address @"__iree_flow___sm_node618__m.layer-79.kernel" : !util.ptr<tensor<1x1x240x64xf32>>
    %79 = flow.variable.address @"__iree_flow___sm_node619__m.layer-79.bias" : !util.ptr<tensor<64xf32>>
    %80 = flow.variable.address @"__iree_flow___sm_node628__m.layer-81.kernel" : !util.ptr<tensor<1x1x64x240xf32>>
    %81 = flow.variable.address @"__iree_flow___sm_node629__m.layer-81.bias" : !util.ptr<tensor<240xf32>>
    %82 = flow.variable.address @"__iree_flow___sm_node644__m.layer-86.kernel" : !util.ptr<tensor<1x1x240x40xf32>>
    %83 = flow.variable.address @"__iree_flow___sm_node650__m.layer-87.gamma" : !util.ptr<tensor<40xf32>>
    %84 = flow.variable.address @"__iree_flow___sm_node651__m.layer-87.beta" : !util.ptr<tensor<40xf32>>
    %85 = flow.variable.address @"__iree_flow___sm_node652__m.layer-87.moving_mean" : !util.ptr<tensor<40xf32>>
    %86 = flow.variable.address @"__iree_flow___sm_node653__m.layer-87.moving_variance" : !util.ptr<tensor<40xf32>>
    %87 = flow.variable.address @"__iree_flow___sm_node662__m.layer-89.kernel" : !util.ptr<tensor<1x1x40x240xf32>>
    %88 = flow.variable.address @"__iree_flow___sm_node668__m.layer-90.gamma" : !util.ptr<tensor<240xf32>>
    %89 = flow.variable.address @"__iree_flow___sm_node669__m.layer-90.beta" : !util.ptr<tensor<240xf32>>
    %90 = flow.variable.address @"__iree_flow___sm_node670__m.layer-90.moving_mean" : !util.ptr<tensor<240xf32>>
    %91 = flow.variable.address @"__iree_flow___sm_node671__m.layer-90.moving_variance" : !util.ptr<tensor<240xf32>>
    %92 = flow.variable.address @"__iree_flow___sm_node686__m.layer-95.depthwise_kernel" : !util.ptr<tensor<5x5x240x1xf32>>
    %93 = flow.variable.address @"__iree_flow___sm_node692__m.layer-96.gamma" : !util.ptr<tensor<240xf32>>
    %94 = flow.variable.address @"__iree_flow___sm_node693__m.layer-96.beta" : !util.ptr<tensor<240xf32>>
    %95 = flow.variable.address @"__iree_flow___sm_node694__m.layer-96.moving_mean" : !util.ptr<tensor<240xf32>>
    %96 = flow.variable.address @"__iree_flow___sm_node695__m.layer-96.moving_variance" : !util.ptr<tensor<240xf32>>
    %97 = flow.variable.address @"__iree_flow___sm_node718__m.layer-103.kernel" : !util.ptr<tensor<1x1x240x64xf32>>
    %98 = flow.variable.address @"__iree_flow___sm_node719__m.layer-103.bias" : !util.ptr<tensor<64xf32>>
    %99 = flow.variable.address @"__iree_flow___sm_node728__m.layer-105.kernel" : !util.ptr<tensor<1x1x64x240xf32>>
    %100 = flow.variable.address @"__iree_flow___sm_node729__m.layer-105.bias" : !util.ptr<tensor<240xf32>>
    %101 = flow.variable.address @"__iree_flow___sm_node744__m.layer-110.kernel" : !util.ptr<tensor<1x1x240x40xf32>>
    %102 = flow.variable.address @"__iree_flow___sm_node750__m.layer-111.gamma" : !util.ptr<tensor<40xf32>>
    %103 = flow.variable.address @"__iree_flow___sm_node751__m.layer-111.beta" : !util.ptr<tensor<40xf32>>
    %104 = flow.variable.address @"__iree_flow___sm_node752__m.layer-111.moving_mean" : !util.ptr<tensor<40xf32>>
    %105 = flow.variable.address @"__iree_flow___sm_node753__m.layer-111.moving_variance" : !util.ptr<tensor<40xf32>>
    %106 = flow.variable.address @"__iree_flow___sm_node762__m.layer-113.kernel" : !util.ptr<tensor<1x1x40x120xf32>>
    %107 = flow.variable.address @"__iree_flow___sm_node768__m.layer-114.gamma" : !util.ptr<tensor<120xf32>>
    %108 = flow.variable.address @"__iree_flow___sm_node769__m.layer-114.beta" : !util.ptr<tensor<120xf32>>
    %109 = flow.variable.address @"__iree_flow___sm_node770__m.layer-114.moving_mean" : !util.ptr<tensor<120xf32>>
    %110 = flow.variable.address @"__iree_flow___sm_node771__m.layer-114.moving_variance" : !util.ptr<tensor<120xf32>>
    %111 = flow.variable.address @"__iree_flow___sm_node786__m.layer-119.depthwise_kernel" : !util.ptr<tensor<5x5x120x1xf32>>
    %112 = flow.variable.address @"__iree_flow___sm_node792__m.layer-120.gamma" : !util.ptr<tensor<120xf32>>
    %113 = flow.variable.address @"__iree_flow___sm_node793__m.layer-120.beta" : !util.ptr<tensor<120xf32>>
    %114 = flow.variable.address @"__iree_flow___sm_node794__m.layer-120.moving_mean" : !util.ptr<tensor<120xf32>>
    %115 = flow.variable.address @"__iree_flow___sm_node795__m.layer-120.moving_variance" : !util.ptr<tensor<120xf32>>
    %116 = flow.variable.address @"__iree_flow___sm_node818__m.layer-127.kernel" : !util.ptr<tensor<1x1x120x32xf32>>
    %117 = flow.variable.address @"__iree_flow___sm_node819__m.layer-127.bias" : !util.ptr<tensor<32xf32>>
    %118 = flow.variable.address @"__iree_flow___sm_node828__m.layer-129.kernel" : !util.ptr<tensor<1x1x32x120xf32>>
    %119 = flow.variable.address @"__iree_flow___sm_node829__m.layer-129.bias" : !util.ptr<tensor<120xf32>>
    %120 = flow.variable.address @"__iree_flow___sm_node844__m.layer-134.kernel" : !util.ptr<tensor<1x1x120x48xf32>>
    %121 = flow.variable.address @"__iree_flow___sm_node850__m.layer-135.gamma" : !util.ptr<tensor<48xf32>>
    %122 = flow.variable.address @"__iree_flow___sm_node851__m.layer-135.beta" : !util.ptr<tensor<48xf32>>
    %123 = flow.variable.address @"__iree_flow___sm_node852__m.layer-135.moving_mean" : !util.ptr<tensor<48xf32>>
    %124 = flow.variable.address @"__iree_flow___sm_node853__m.layer-135.moving_variance" : !util.ptr<tensor<48xf32>>
    %125 = flow.variable.address @"__iree_flow___sm_node858__m.layer-136.kernel" : !util.ptr<tensor<1x1x48x144xf32>>
    %126 = flow.variable.address @"__iree_flow___sm_node864__m.layer-137.gamma" : !util.ptr<tensor<144xf32>>
    %127 = flow.variable.address @"__iree_flow___sm_node865__m.layer-137.beta" : !util.ptr<tensor<144xf32>>
    %128 = flow.variable.address @"__iree_flow___sm_node866__m.layer-137.moving_mean" : !util.ptr<tensor<144xf32>>
    %129 = flow.variable.address @"__iree_flow___sm_node867__m.layer-137.moving_variance" : !util.ptr<tensor<144xf32>>
    %130 = flow.variable.address @"__iree_flow___sm_node882__m.layer-142.depthwise_kernel" : !util.ptr<tensor<5x5x144x1xf32>>
    %131 = flow.variable.address @"__iree_flow___sm_node888__m.layer-143.gamma" : !util.ptr<tensor<144xf32>>
    %132 = flow.variable.address @"__iree_flow___sm_node889__m.layer-143.beta" : !util.ptr<tensor<144xf32>>
    %133 = flow.variable.address @"__iree_flow___sm_node890__m.layer-143.moving_mean" : !util.ptr<tensor<144xf32>>
    %134 = flow.variable.address @"__iree_flow___sm_node891__m.layer-143.moving_variance" : !util.ptr<tensor<144xf32>>
    %135 = flow.variable.address @"__iree_flow___sm_node914__m.layer-150.kernel" : !util.ptr<tensor<1x1x144x40xf32>>
    %136 = flow.variable.address @"__iree_flow___sm_node915__m.layer-150.bias" : !util.ptr<tensor<40xf32>>
    %137 = flow.variable.address @"__iree_flow___sm_node924__m.layer-152.kernel" : !util.ptr<tensor<1x1x40x144xf32>>
    %138 = flow.variable.address @"__iree_flow___sm_node925__m.layer-152.bias" : !util.ptr<tensor<144xf32>>
    %139 = flow.variable.address @"__iree_flow___sm_node940__m.layer-157.kernel" : !util.ptr<tensor<1x1x144x48xf32>>
    %140 = flow.variable.address @"__iree_flow___sm_node946__m.layer-158.gamma" : !util.ptr<tensor<48xf32>>
    %141 = flow.variable.address @"__iree_flow___sm_node947__m.layer-158.beta" : !util.ptr<tensor<48xf32>>
    %142 = flow.variable.address @"__iree_flow___sm_node948__m.layer-158.moving_mean" : !util.ptr<tensor<48xf32>>
    %143 = flow.variable.address @"__iree_flow___sm_node949__m.layer-158.moving_variance" : !util.ptr<tensor<48xf32>>
    %144 = flow.variable.address @"__iree_flow___sm_node958__m.layer-160.kernel" : !util.ptr<tensor<1x1x48x288xf32>>
    %145 = flow.variable.address @"__iree_flow___sm_node964__m.layer-161.gamma" : !util.ptr<tensor<288xf32>>
    %146 = flow.variable.address @"__iree_flow___sm_node965__m.layer-161.beta" : !util.ptr<tensor<288xf32>>
    %147 = flow.variable.address @"__iree_flow___sm_node966__m.layer-161.moving_mean" : !util.ptr<tensor<288xf32>>
    %148 = flow.variable.address @"__iree_flow___sm_node967__m.layer-161.moving_variance" : !util.ptr<tensor<288xf32>>
    %149 = flow.variable.address @"__iree_flow___sm_node986__m.layer-167.depthwise_kernel" : !util.ptr<tensor<5x5x288x1xf32>>
    %150 = flow.variable.address @"__iree_flow___sm_node992__m.layer-168.gamma" : !util.ptr<tensor<288xf32>>
    %151 = flow.variable.address @"__iree_flow___sm_node993__m.layer-168.beta" : !util.ptr<tensor<288xf32>>
    %152 = flow.variable.address @"__iree_flow___sm_node994__m.layer-168.moving_mean" : !util.ptr<tensor<288xf32>>
    %153 = flow.variable.address @"__iree_flow___sm_node995__m.layer-168.moving_variance" : !util.ptr<tensor<288xf32>>
    %154 = flow.variable.address @"__iree_flow___sm_node1018__m.layer-175.kernel" : !util.ptr<tensor<1x1x288x72xf32>>
    %155 = flow.variable.address @"__iree_flow___sm_node1019__m.layer-175.bias" : !util.ptr<tensor<72xf32>>
    %156 = flow.variable.address @"__iree_flow___sm_node1028__m.layer-177.kernel" : !util.ptr<tensor<1x1x72x288xf32>>
    %157 = flow.variable.address @"__iree_flow___sm_node1029__m.layer-177.bias" : !util.ptr<tensor<288xf32>>
    %158 = flow.variable.address @"__iree_flow___sm_node1044__m.layer-182.kernel" : !util.ptr<tensor<1x1x288x96xf32>>
    %159 = flow.variable.address @"__iree_flow___sm_node1050__m.layer-183.gamma" : !util.ptr<tensor<96xf32>>
    %160 = flow.variable.address @"__iree_flow___sm_node1051__m.layer-183.beta" : !util.ptr<tensor<96xf32>>
    %161 = flow.variable.address @"__iree_flow___sm_node1052__m.layer-183.moving_mean" : !util.ptr<tensor<96xf32>>
    %162 = flow.variable.address @"__iree_flow___sm_node1053__m.layer-183.moving_variance" : !util.ptr<tensor<96xf32>>
    %163 = flow.variable.address @"__iree_flow___sm_node1058__m.layer-184.kernel" : !util.ptr<tensor<1x1x96x576xf32>>
    %164 = flow.variable.address @"__iree_flow___sm_node1064__m.layer-185.gamma" : !util.ptr<tensor<576xf32>>
    %165 = flow.variable.address @"__iree_flow___sm_node1065__m.layer-185.beta" : !util.ptr<tensor<576xf32>>
    %166 = flow.variable.address @"__iree_flow___sm_node1066__m.layer-185.moving_mean" : !util.ptr<tensor<576xf32>>
    %167 = flow.variable.address @"__iree_flow___sm_node1067__m.layer-185.moving_variance" : !util.ptr<tensor<576xf32>>
    %168 = flow.variable.address @"__iree_flow___sm_node1082__m.layer-190.depthwise_kernel" : !util.ptr<tensor<5x5x576x1xf32>>
    %169 = flow.variable.address @"__iree_flow___sm_node1088__m.layer-191.gamma" : !util.ptr<tensor<576xf32>>
    %170 = flow.variable.address @"__iree_flow___sm_node1089__m.layer-191.beta" : !util.ptr<tensor<576xf32>>
    %171 = flow.variable.address @"__iree_flow___sm_node1090__m.layer-191.moving_mean" : !util.ptr<tensor<576xf32>>
    %172 = flow.variable.address @"__iree_flow___sm_node1091__m.layer-191.moving_variance" : !util.ptr<tensor<576xf32>>
    %173 = flow.variable.address @"__iree_flow___sm_node1114__m.layer-198.kernel" : !util.ptr<tensor<1x1x576x144xf32>>
    %174 = flow.variable.address @"__iree_flow___sm_node1115__m.layer-198.bias" : !util.ptr<tensor<144xf32>>
    %175 = flow.variable.address @"__iree_flow___sm_node1124__m.layer-200.kernel" : !util.ptr<tensor<1x1x144x576xf32>>
    %176 = flow.variable.address @"__iree_flow___sm_node1125__m.layer-200.bias" : !util.ptr<tensor<576xf32>>
    %177 = flow.variable.address @"__iree_flow___sm_node1140__m.layer-205.kernel" : !util.ptr<tensor<1x1x576x96xf32>>
    %178 = flow.variable.address @"__iree_flow___sm_node1146__m.layer-206.gamma" : !util.ptr<tensor<96xf32>>
    %179 = flow.variable.address @"__iree_flow___sm_node1147__m.layer-206.beta" : !util.ptr<tensor<96xf32>>
    %180 = flow.variable.address @"__iree_flow___sm_node1148__m.layer-206.moving_mean" : !util.ptr<tensor<96xf32>>
    %181 = flow.variable.address @"__iree_flow___sm_node1149__m.layer-206.moving_variance" : !util.ptr<tensor<96xf32>>
    %182 = flow.variable.address @"__iree_flow___sm_node1158__m.layer-208.kernel" : !util.ptr<tensor<1x1x96x576xf32>>
    %183 = flow.variable.address @"__iree_flow___sm_node1164__m.layer-209.gamma" : !util.ptr<tensor<576xf32>>
    %184 = flow.variable.address @"__iree_flow___sm_node1165__m.layer-209.beta" : !util.ptr<tensor<576xf32>>
    %185 = flow.variable.address @"__iree_flow___sm_node1166__m.layer-209.moving_mean" : !util.ptr<tensor<576xf32>>
    %186 = flow.variable.address @"__iree_flow___sm_node1167__m.layer-209.moving_variance" : !util.ptr<tensor<576xf32>>
    %187 = flow.variable.address @"__iree_flow___sm_node1182__m.layer-214.depthwise_kernel" : !util.ptr<tensor<5x5x576x1xf32>>
    %188 = flow.variable.address @"__iree_flow___sm_node1188__m.layer-215.gamma" : !util.ptr<tensor<576xf32>>
    %189 = flow.variable.address @"__iree_flow___sm_node1189__m.layer-215.beta" : !util.ptr<tensor<576xf32>>
    %190 = flow.variable.address @"__iree_flow___sm_node1190__m.layer-215.moving_mean" : !util.ptr<tensor<576xf32>>
    %191 = flow.variable.address @"__iree_flow___sm_node1191__m.layer-215.moving_variance" : !util.ptr<tensor<576xf32>>
    %192 = flow.variable.address @"__iree_flow___sm_node1214__m.layer-222.kernel" : !util.ptr<tensor<1x1x576x144xf32>>
    %193 = flow.variable.address @"__iree_flow___sm_node1215__m.layer-222.bias" : !util.ptr<tensor<144xf32>>
    %194 = flow.variable.address @"__iree_flow___sm_node1224__m.layer-224.kernel" : !util.ptr<tensor<1x1x144x576xf32>>
    %195 = flow.variable.address @"__iree_flow___sm_node1225__m.layer-224.bias" : !util.ptr<tensor<576xf32>>
    %196 = flow.variable.address @"__iree_flow___sm_node1240__m.layer-229.kernel" : !util.ptr<tensor<1x1x576x96xf32>>
    %197 = flow.variable.address @"__iree_flow___sm_node1246__m.layer-230.gamma" : !util.ptr<tensor<96xf32>>
    %198 = flow.variable.address @"__iree_flow___sm_node1247__m.layer-230.beta" : !util.ptr<tensor<96xf32>>
    %199 = flow.variable.address @"__iree_flow___sm_node1248__m.layer-230.moving_mean" : !util.ptr<tensor<96xf32>>
    %200 = flow.variable.address @"__iree_flow___sm_node1249__m.layer-230.moving_variance" : !util.ptr<tensor<96xf32>>
    %201 = flow.variable.address @"__iree_flow___sm_node1258__m.layer-232.kernel" : !util.ptr<tensor<1x1x96x576xf32>>
    %202 = flow.variable.address @"__iree_flow___sm_node1264__m.layer-233.gamma" : !util.ptr<tensor<576xf32>>
    %203 = flow.variable.address @"__iree_flow___sm_node1265__m.layer-233.beta" : !util.ptr<tensor<576xf32>>
    %204 = flow.variable.address @"__iree_flow___sm_node1266__m.layer-233.moving_mean" : !util.ptr<tensor<576xf32>>
    %205 = flow.variable.address @"__iree_flow___sm_node1267__m.layer-233.moving_variance" : !util.ptr<tensor<576xf32>>
    %206 = flow.variable.address @"__iree_flow___sm_node1290__m.layer-240.kernel" : !util.ptr<tensor<1x1x576x1024xf32>>
    %207 = flow.variable.address @"__iree_flow___sm_node1291__m.layer-240.bias" : !util.ptr<tensor<1024xf32>>
    %208 = flow.variable.address @"__iree_flow___sm_node1310__m.layer-246.kernel" : !util.ptr<tensor<1x1x1024x1000xf32>>
    %209 = flow.variable.address @"__iree_flow___sm_node1311__m.layer-246.bias" : !util.ptr<tensor<1000xf32>>
    %210 = mhlo.constant dense<0.00784313772> : tensor<1x224x224x3xf32>
    %211 = mhlo.constant dense<-1.000000e+00> : tensor<1x224x224x3xf32>
    %212 = mhlo.constant dense<3.000000e+00> : tensor<1x112x112x16xf32>
    %213 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x16xf32>
    %214 = mhlo.constant dense<3.000000e+00> : tensor<1x28x28x96xf32>
    %215 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x96xf32>
    %216 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x96xf32>
    %217 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x240xf32>
    %218 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x240xf32>
    %219 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x120xf32>
    %220 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x120xf32>
    %221 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x144xf32>
    %222 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x144xf32>
    %223 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x288xf32>
    %224 = mhlo.constant dense<3.000000e+00> : tensor<1x7x7x288xf32>
    %225 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x288xf32>
    %226 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x576xf32>
    %227 = mhlo.constant dense<3.000000e+00> : tensor<1x7x7x576xf32>
    %228 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x1024xf32>
    %229 = mhlo.constant dense<0.166666672> : tensor<1x112x112x16xf32>
    %230 = mhlo.constant dense<0.166666672> : tensor<1x1x1x16xf32>
    %231 = mhlo.constant dense<0.166666672> : tensor<1x28x28x96xf32>
    %232 = mhlo.constant dense<0.166666672> : tensor<1x14x14x96xf32>
    %233 = mhlo.constant dense<0.166666672> : tensor<1x1x1x96xf32>
    %234 = mhlo.constant dense<0.166666672> : tensor<1x14x14x240xf32>
    %235 = mhlo.constant dense<0.166666672> : tensor<1x1x1x240xf32>
    %236 = mhlo.constant dense<0.166666672> : tensor<1x14x14x120xf32>
    %237 = mhlo.constant dense<0.166666672> : tensor<1x1x1x120xf32>
    %238 = mhlo.constant dense<0.166666672> : tensor<1x14x14x144xf32>
    %239 = mhlo.constant dense<0.166666672> : tensor<1x1x1x144xf32>
    %240 = mhlo.constant dense<0.166666672> : tensor<1x14x14x288xf32>
    %241 = mhlo.constant dense<0.166666672> : tensor<1x7x7x288xf32>
    %242 = mhlo.constant dense<0.166666672> : tensor<1x1x1x288xf32>
    %243 = mhlo.constant dense<0.166666672> : tensor<1x1x1x576xf32>
    %244 = mhlo.constant dense<0.166666672> : tensor<1x7x7x576xf32>
    %245 = mhlo.constant dense<0.166666672> : tensor<1x1x1x1024xf32>
    %246 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x16xf32>
    %247 = mhlo.constant dense<3.136000e+03> : tensor<1x16xf32>
    %248 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x8xf32>
    %249 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x72xf32>
    %250 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x72xf32>
    %251 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x88xf32>
    %252 = mhlo.constant dense<1.960000e+02> : tensor<1x96xf32>
    %253 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x24xf32>
    %254 = mhlo.constant dense<1.960000e+02> : tensor<1x240xf32>
    %255 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x64xf32>
    %256 = mhlo.constant dense<1.960000e+02> : tensor<1x120xf32>
    %257 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x32xf32>
    %258 = mhlo.constant dense<1.960000e+02> : tensor<1x144xf32>
    %259 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x40xf32>
    %260 = mhlo.constant dense<4.900000e+01> : tensor<1x288xf32>
    %261 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x72xf32>
    %262 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x144xf32>
    %263 = mhlo.constant dense<4.900000e+01> : tensor<1x576xf32>
    %264 = mhlo.constant dense<6.000000e+00> : tensor<f32>
    %265 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %266 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %267 = flow.variable.load.indirect %205 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %268 = flow.variable.load.indirect %204 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %269 = flow.variable.load.indirect %203 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %270 = flow.variable.load.indirect %202 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %271 = flow.variable.load.indirect %201 : !util.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %272 = flow.variable.load.indirect %207 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
    %273 = flow.variable.load.indirect %206 : !util.ptr<tensor<1x1x576x1024xf32>> -> tensor<1x1x576x1024xf32>
    %274 = flow.variable.load.indirect %4 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %275 = flow.variable.load.indirect %3 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %276 = flow.variable.load.indirect %2 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %277 = flow.variable.load.indirect %1 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %278 = flow.variable.load.indirect %0 : !util.ptr<tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
    %279 = flow.variable.load.indirect %191 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %280 = flow.variable.load.indirect %190 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %281 = flow.variable.load.indirect %189 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %282 = flow.variable.load.indirect %188 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %283 = flow.variable.load.indirect %187 : !util.ptr<tensor<5x5x576x1xf32>> -> tensor<5x5x576x1xf32>
    %284 = flow.variable.load.indirect %186 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %285 = flow.variable.load.indirect %185 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %286 = flow.variable.load.indirect %184 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %287 = flow.variable.load.indirect %183 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %288 = flow.variable.load.indirect %182 : !util.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %289 = flow.variable.load.indirect %200 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %290 = flow.variable.load.indirect %199 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %291 = flow.variable.load.indirect %198 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %292 = flow.variable.load.indirect %197 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %293 = flow.variable.load.indirect %196 : !util.ptr<tensor<1x1x576x96xf32>> -> tensor<1x1x576x96xf32>
    %294 = flow.variable.load.indirect %195 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %295 = flow.variable.load.indirect %194 : !util.ptr<tensor<1x1x144x576xf32>> -> tensor<1x1x144x576xf32>
    %296 = flow.variable.load.indirect %193 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %297 = flow.variable.load.indirect %192 : !util.ptr<tensor<1x1x576x144xf32>> -> tensor<1x1x576x144xf32>
    %298 = flow.variable.load.indirect %28 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %299 = flow.variable.load.indirect %27 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %300 = flow.variable.load.indirect %26 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %301 = flow.variable.load.indirect %25 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %302 = flow.variable.load.indirect %24 : !util.ptr<tensor<3x3x72x1xf32>> -> tensor<3x3x72x1xf32>
    %303 = flow.variable.load.indirect %23 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %304 = flow.variable.load.indirect %22 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %305 = flow.variable.load.indirect %21 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %306 = flow.variable.load.indirect %20 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %307 = flow.variable.load.indirect %19 : !util.ptr<tensor<1x1x16x72xf32>> -> tensor<1x1x16x72xf32>
    %308 = flow.variable.load.indirect %33 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %309 = flow.variable.load.indirect %32 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %310 = flow.variable.load.indirect %31 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %311 = flow.variable.load.indirect %30 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %312 = flow.variable.load.indirect %29 : !util.ptr<tensor<1x1x72x24xf32>> -> tensor<1x1x72x24xf32>
    %313 = flow.variable.load.indirect %43 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %314 = flow.variable.load.indirect %42 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %315 = flow.variable.load.indirect %41 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %316 = flow.variable.load.indirect %40 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %317 = flow.variable.load.indirect %39 : !util.ptr<tensor<3x3x88x1xf32>> -> tensor<3x3x88x1xf32>
    %318 = flow.variable.load.indirect %38 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %319 = flow.variable.load.indirect %37 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %320 = flow.variable.load.indirect %36 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %321 = flow.variable.load.indirect %35 : !util.ptr<tensor<88xf32>> -> tensor<88xf32>
    %322 = flow.variable.load.indirect %34 : !util.ptr<tensor<1x1x24x88xf32>> -> tensor<1x1x24x88xf32>
    %323 = flow.variable.load.indirect %48 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %324 = flow.variable.load.indirect %47 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %325 = flow.variable.load.indirect %46 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %326 = flow.variable.load.indirect %45 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %327 = flow.variable.load.indirect %44 : !util.ptr<tensor<1x1x88x24xf32>> -> tensor<1x1x88x24xf32>
    %328 = flow.variable.load.indirect %58 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %329 = flow.variable.load.indirect %57 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %330 = flow.variable.load.indirect %56 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %331 = flow.variable.load.indirect %55 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %332 = flow.variable.load.indirect %54 : !util.ptr<tensor<5x5x96x1xf32>> -> tensor<5x5x96x1xf32>
    %333 = flow.variable.load.indirect %53 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %334 = flow.variable.load.indirect %52 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %335 = flow.variable.load.indirect %51 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %336 = flow.variable.load.indirect %50 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %337 = flow.variable.load.indirect %49 : !util.ptr<tensor<1x1x24x96xf32>> -> tensor<1x1x24x96xf32>
    %338 = flow.variable.load.indirect %67 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %339 = flow.variable.load.indirect %66 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %340 = flow.variable.load.indirect %65 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %341 = flow.variable.load.indirect %64 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %342 = flow.variable.load.indirect %63 : !util.ptr<tensor<1x1x96x40xf32>> -> tensor<1x1x96x40xf32>
    %343 = flow.variable.load.indirect %62 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %344 = flow.variable.load.indirect %61 : !util.ptr<tensor<1x1x24x96xf32>> -> tensor<1x1x24x96xf32>
    %345 = flow.variable.load.indirect %60 : !util.ptr<tensor<24xf32>> -> tensor<24xf32>
    %346 = flow.variable.load.indirect %59 : !util.ptr<tensor<1x1x96x24xf32>> -> tensor<1x1x96x24xf32>
    %347 = flow.variable.load.indirect %77 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %348 = flow.variable.load.indirect %76 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %349 = flow.variable.load.indirect %75 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %350 = flow.variable.load.indirect %74 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %351 = flow.variable.load.indirect %73 : !util.ptr<tensor<5x5x240x1xf32>> -> tensor<5x5x240x1xf32>
    %352 = flow.variable.load.indirect %72 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %353 = flow.variable.load.indirect %71 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %354 = flow.variable.load.indirect %70 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %355 = flow.variable.load.indirect %69 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %356 = flow.variable.load.indirect %68 : !util.ptr<tensor<1x1x40x240xf32>> -> tensor<1x1x40x240xf32>
    %357 = flow.variable.load.indirect %86 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %358 = flow.variable.load.indirect %85 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %359 = flow.variable.load.indirect %84 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %360 = flow.variable.load.indirect %83 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %361 = flow.variable.load.indirect %82 : !util.ptr<tensor<1x1x240x40xf32>> -> tensor<1x1x240x40xf32>
    %362 = flow.variable.load.indirect %81 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %363 = flow.variable.load.indirect %80 : !util.ptr<tensor<1x1x64x240xf32>> -> tensor<1x1x64x240xf32>
    %364 = flow.variable.load.indirect %79 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
    %365 = flow.variable.load.indirect %78 : !util.ptr<tensor<1x1x240x64xf32>> -> tensor<1x1x240x64xf32>
    %366 = flow.variable.load.indirect %96 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %367 = flow.variable.load.indirect %95 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %368 = flow.variable.load.indirect %94 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %369 = flow.variable.load.indirect %93 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %370 = flow.variable.load.indirect %92 : !util.ptr<tensor<5x5x240x1xf32>> -> tensor<5x5x240x1xf32>
    %371 = flow.variable.load.indirect %91 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %372 = flow.variable.load.indirect %90 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %373 = flow.variable.load.indirect %89 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %374 = flow.variable.load.indirect %88 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %375 = flow.variable.load.indirect %87 : !util.ptr<tensor<1x1x40x240xf32>> -> tensor<1x1x40x240xf32>
    %376 = flow.variable.load.indirect %105 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %377 = flow.variable.load.indirect %104 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %378 = flow.variable.load.indirect %103 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %379 = flow.variable.load.indirect %102 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %380 = flow.variable.load.indirect %101 : !util.ptr<tensor<1x1x240x40xf32>> -> tensor<1x1x240x40xf32>
    %381 = flow.variable.load.indirect %100 : !util.ptr<tensor<240xf32>> -> tensor<240xf32>
    %382 = flow.variable.load.indirect %99 : !util.ptr<tensor<1x1x64x240xf32>> -> tensor<1x1x64x240xf32>
    %383 = flow.variable.load.indirect %98 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
    %384 = flow.variable.load.indirect %97 : !util.ptr<tensor<1x1x240x64xf32>> -> tensor<1x1x240x64xf32>
    %385 = flow.variable.load.indirect %115 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %386 = flow.variable.load.indirect %114 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %387 = flow.variable.load.indirect %113 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %388 = flow.variable.load.indirect %112 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %389 = flow.variable.load.indirect %111 : !util.ptr<tensor<5x5x120x1xf32>> -> tensor<5x5x120x1xf32>
    %390 = flow.variable.load.indirect %110 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %391 = flow.variable.load.indirect %109 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %392 = flow.variable.load.indirect %108 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %393 = flow.variable.load.indirect %107 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %394 = flow.variable.load.indirect %106 : !util.ptr<tensor<1x1x40x120xf32>> -> tensor<1x1x40x120xf32>
    %395 = flow.variable.load.indirect %124 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %396 = flow.variable.load.indirect %123 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %397 = flow.variable.load.indirect %122 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %398 = flow.variable.load.indirect %121 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %399 = flow.variable.load.indirect %120 : !util.ptr<tensor<1x1x120x48xf32>> -> tensor<1x1x120x48xf32>
    %400 = flow.variable.load.indirect %119 : !util.ptr<tensor<120xf32>> -> tensor<120xf32>
    %401 = flow.variable.load.indirect %118 : !util.ptr<tensor<1x1x32x120xf32>> -> tensor<1x1x32x120xf32>
    %402 = flow.variable.load.indirect %117 : !util.ptr<tensor<32xf32>> -> tensor<32xf32>
    %403 = flow.variable.load.indirect %116 : !util.ptr<tensor<1x1x120x32xf32>> -> tensor<1x1x120x32xf32>
    %404 = flow.variable.load.indirect %134 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %405 = flow.variable.load.indirect %133 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %406 = flow.variable.load.indirect %132 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %407 = flow.variable.load.indirect %131 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %408 = flow.variable.load.indirect %130 : !util.ptr<tensor<5x5x144x1xf32>> -> tensor<5x5x144x1xf32>
    %409 = flow.variable.load.indirect %129 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %410 = flow.variable.load.indirect %128 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %411 = flow.variable.load.indirect %127 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %412 = flow.variable.load.indirect %126 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %413 = flow.variable.load.indirect %125 : !util.ptr<tensor<1x1x48x144xf32>> -> tensor<1x1x48x144xf32>
    %414 = flow.variable.load.indirect %143 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %415 = flow.variable.load.indirect %142 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %416 = flow.variable.load.indirect %141 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %417 = flow.variable.load.indirect %140 : !util.ptr<tensor<48xf32>> -> tensor<48xf32>
    %418 = flow.variable.load.indirect %139 : !util.ptr<tensor<1x1x144x48xf32>> -> tensor<1x1x144x48xf32>
    %419 = flow.variable.load.indirect %138 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %420 = flow.variable.load.indirect %137 : !util.ptr<tensor<1x1x40x144xf32>> -> tensor<1x1x40x144xf32>
    %421 = flow.variable.load.indirect %136 : !util.ptr<tensor<40xf32>> -> tensor<40xf32>
    %422 = flow.variable.load.indirect %135 : !util.ptr<tensor<1x1x144x40xf32>> -> tensor<1x1x144x40xf32>
    %423 = flow.variable.load.indirect %153 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %424 = flow.variable.load.indirect %152 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %425 = flow.variable.load.indirect %151 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %426 = flow.variable.load.indirect %150 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %427 = flow.variable.load.indirect %149 : !util.ptr<tensor<5x5x288x1xf32>> -> tensor<5x5x288x1xf32>
    %428 = flow.variable.load.indirect %148 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %429 = flow.variable.load.indirect %147 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %430 = flow.variable.load.indirect %146 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %431 = flow.variable.load.indirect %145 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %432 = flow.variable.load.indirect %144 : !util.ptr<tensor<1x1x48x288xf32>> -> tensor<1x1x48x288xf32>
    %433 = flow.variable.load.indirect %162 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %434 = flow.variable.load.indirect %161 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %435 = flow.variable.load.indirect %160 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %436 = flow.variable.load.indirect %159 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %437 = flow.variable.load.indirect %158 : !util.ptr<tensor<1x1x288x96xf32>> -> tensor<1x1x288x96xf32>
    %438 = flow.variable.load.indirect %157 : !util.ptr<tensor<288xf32>> -> tensor<288xf32>
    %439 = flow.variable.load.indirect %156 : !util.ptr<tensor<1x1x72x288xf32>> -> tensor<1x1x72x288xf32>
    %440 = flow.variable.load.indirect %155 : !util.ptr<tensor<72xf32>> -> tensor<72xf32>
    %441 = flow.variable.load.indirect %154 : !util.ptr<tensor<1x1x288x72xf32>> -> tensor<1x1x288x72xf32>
    %442 = flow.variable.load.indirect %172 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %443 = flow.variable.load.indirect %171 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %444 = flow.variable.load.indirect %170 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %445 = flow.variable.load.indirect %169 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %446 = flow.variable.load.indirect %168 : !util.ptr<tensor<5x5x576x1xf32>> -> tensor<5x5x576x1xf32>
    %447 = flow.variable.load.indirect %167 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %448 = flow.variable.load.indirect %166 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %449 = flow.variable.load.indirect %165 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %450 = flow.variable.load.indirect %164 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %451 = flow.variable.load.indirect %163 : !util.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %452 = flow.variable.load.indirect %181 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %453 = flow.variable.load.indirect %180 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %454 = flow.variable.load.indirect %179 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %455 = flow.variable.load.indirect %178 : !util.ptr<tensor<96xf32>> -> tensor<96xf32>
    %456 = flow.variable.load.indirect %177 : !util.ptr<tensor<1x1x576x96xf32>> -> tensor<1x1x576x96xf32>
    %457 = flow.variable.load.indirect %176 : !util.ptr<tensor<576xf32>> -> tensor<576xf32>
    %458 = flow.variable.load.indirect %175 : !util.ptr<tensor<1x1x144x576xf32>> -> tensor<1x1x144x576xf32>
    %459 = flow.variable.load.indirect %174 : !util.ptr<tensor<144xf32>> -> tensor<144xf32>
    %460 = flow.variable.load.indirect %173 : !util.ptr<tensor<1x1x576x144xf32>> -> tensor<1x1x576x144xf32>
    %461 = flow.variable.load.indirect %9 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %462 = flow.variable.load.indirect %8 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %463 = flow.variable.load.indirect %7 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %464 = flow.variable.load.indirect %6 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %465 = flow.variable.load.indirect %5 : !util.ptr<tensor<3x3x16x1xf32>> -> tensor<3x3x16x1xf32>
    %466 = flow.variable.load.indirect %18 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %467 = flow.variable.load.indirect %17 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %468 = flow.variable.load.indirect %16 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %469 = flow.variable.load.indirect %15 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %470 = flow.variable.load.indirect %14 : !util.ptr<tensor<1x1x16x16xf32>> -> tensor<1x1x16x16xf32>
    %471 = flow.variable.load.indirect %13 : !util.ptr<tensor<16xf32>> -> tensor<16xf32>
    %472 = flow.variable.load.indirect %12 : !util.ptr<tensor<1x1x8x16xf32>> -> tensor<1x1x8x16xf32>
    %473 = flow.variable.load.indirect %11 : !util.ptr<tensor<8xf32>> -> tensor<8xf32>
    %474 = flow.variable.load.indirect %10 : !util.ptr<tensor<1x1x16x8xf32>> -> tensor<1x1x16x8xf32>
    %475 = flow.variable.load.indirect %209 : !util.ptr<tensor<1000xf32>> -> tensor<1000xf32>
    %476 = flow.variable.load.indirect %208 : !util.ptr<tensor<1x1x1024x1000xf32>> -> tensor<1x1x1024x1000xf32>
    %477 = mhlo.multiply %arg0, %210 : tensor<1x224x224x3xf32>
    %478 = mhlo.add %477, %211 : tensor<1x224x224x3xf32>
    %479 = "mhlo.convolution"(%478, %278) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x224x224x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x112x112x16xf32>
    %480 = "mhlo.batch_norm_inference"(%479, %277, %276, %275, %274) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %481 = mhlo.add %480, %212 : tensor<1x112x112x16xf32>
    %482 = "mhlo.clamp"(%266, %481, %264) : (tensor<f32>, tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
    %483 = mhlo.multiply %482, %229 : tensor<1x112x112x16xf32>
    %484 = mhlo.multiply %483, %480 : tensor<1x112x112x16xf32>
    %485 = "mhlo.pad"(%484, %266) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x113x113x16xf32>
    %486 = "mhlo.reshape"(%465) : (tensor<3x3x16x1xf32>) -> tensor<3x3x1x16xf32>
    %487 = "mhlo.convolution"(%485, %486) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 16 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x16xf32>, tensor<3x3x1x16xf32>) -> tensor<1x56x56x16xf32>
    %488 = "mhlo.batch_norm_inference"(%487, %464, %463, %462, %461) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %489 = mhlo.maximum %488, %246 : tensor<1x56x56x16xf32>
    %490 = "mhlo.reduce"(%489, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<f32>) -> tensor<1x16xf32>
    %491 = mhlo.divide %490, %247 : tensor<1x16xf32>
    %492 = "mhlo.reshape"(%491) : (tensor<1x16xf32>) -> tensor<1x1x1x16xf32>
    %493 = "mhlo.convolution"(%492, %474) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x16xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x1x8xf32>
    %494 = "mhlo.broadcast_in_dim"(%473) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>) -> tensor<1x1x1x8xf32>
    %495 = mhlo.add %493, %494 : tensor<1x1x1x8xf32>
    %496 = mhlo.maximum %495, %248 : tensor<1x1x1x8xf32>
    %497 = "mhlo.convolution"(%496, %472) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x8xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
    %498 = "mhlo.broadcast_in_dim"(%471) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %499 = mhlo.add %497, %498 : tensor<1x1x1x16xf32>
    %500 = mhlo.add %499, %213 : tensor<1x1x1x16xf32>
    %501 = "mhlo.clamp"(%266, %500, %264) : (tensor<f32>, tensor<1x1x1x16xf32>, tensor<f32>) -> tensor<1x1x1x16xf32>
    %502 = mhlo.multiply %501, %230 : tensor<1x1x1x16xf32>
    %503 = "mhlo.broadcast_in_dim"(%502) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x16xf32>) -> tensor<1x56x56x16xf32>
    %504 = mhlo.multiply %489, %503 : tensor<1x56x56x16xf32>
    %505 = "mhlo.convolution"(%504, %470) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x56x56x16xf32>
    %506 = "mhlo.batch_norm_inference"(%505, %469, %468, %467, %466) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %507 = "mhlo.convolution"(%506, %307) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<1x1x16x72xf32>) -> tensor<1x56x56x72xf32>
    %508 = "mhlo.batch_norm_inference"(%507, %306, %305, %304, %303) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> tensor<1x56x56x72xf32>
    %509 = mhlo.maximum %508, %249 : tensor<1x56x56x72xf32>
    %510 = "mhlo.pad"(%509, %266) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x56x56x72xf32>, tensor<f32>) -> tensor<1x57x57x72xf32>
    %511 = "mhlo.reshape"(%302) : (tensor<3x3x72x1xf32>) -> tensor<3x3x1x72xf32>
    %512 = "mhlo.convolution"(%510, %511) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 72 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x57x57x72xf32>, tensor<3x3x1x72xf32>) -> tensor<1x28x28x72xf32>
    %513 = "mhlo.batch_norm_inference"(%512, %301, %300, %299, %298) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> tensor<1x28x28x72xf32>
    %514 = mhlo.maximum %513, %250 : tensor<1x28x28x72xf32>
    %515 = "mhlo.convolution"(%514, %312) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x72xf32>, tensor<1x1x72x24xf32>) -> tensor<1x28x28x24xf32>
    %516 = "mhlo.batch_norm_inference"(%515, %311, %310, %309, %308) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %517 = "mhlo.convolution"(%516, %322) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x24xf32>, tensor<1x1x24x88xf32>) -> tensor<1x28x28x88xf32>
    %518 = "mhlo.batch_norm_inference"(%517, %321, %320, %319, %318) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %519 = mhlo.maximum %518, %251 : tensor<1x28x28x88xf32>
    %520 = "mhlo.reshape"(%317) : (tensor<3x3x88x1xf32>) -> tensor<3x3x1x88xf32>
    %521 = "mhlo.convolution"(%519, %520) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 88 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x88xf32>, tensor<3x3x1x88xf32>) -> tensor<1x28x28x88xf32>
    %522 = "mhlo.batch_norm_inference"(%521, %316, %315, %314, %313) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %523 = mhlo.maximum %522, %251 : tensor<1x28x28x88xf32>
    %524 = "mhlo.convolution"(%523, %327) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x88xf32>, tensor<1x1x88x24xf32>) -> tensor<1x28x28x24xf32>
    %525 = "mhlo.batch_norm_inference"(%524, %326, %325, %324, %323) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %526 = mhlo.add %516, %525 : tensor<1x28x28x24xf32>
    %527 = "mhlo.convolution"(%526, %337) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x24xf32>, tensor<1x1x24x96xf32>) -> tensor<1x28x28x96xf32>
    %528 = "mhlo.batch_norm_inference"(%527, %336, %335, %334, %333) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x28x28x96xf32>
    %529 = mhlo.add %528, %214 : tensor<1x28x28x96xf32>
    %530 = "mhlo.clamp"(%266, %529, %264) : (tensor<f32>, tensor<1x28x28x96xf32>, tensor<f32>) -> tensor<1x28x28x96xf32>
    %531 = mhlo.multiply %530, %231 : tensor<1x28x28x96xf32>
    %532 = mhlo.multiply %531, %528 : tensor<1x28x28x96xf32>
    %533 = "mhlo.pad"(%532, %266) {edge_padding_high = dense<[0, 2, 2, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x28x28x96xf32>, tensor<f32>) -> tensor<1x31x31x96xf32>
    %534 = "mhlo.reshape"(%332) : (tensor<5x5x96x1xf32>) -> tensor<5x5x1x96xf32>
    %535 = "mhlo.convolution"(%533, %534) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 96 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x31x31x96xf32>, tensor<5x5x1x96xf32>) -> tensor<1x14x14x96xf32>
    %536 = "mhlo.batch_norm_inference"(%535, %331, %330, %329, %328) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %537 = mhlo.add %536, %215 : tensor<1x14x14x96xf32>
    %538 = "mhlo.clamp"(%266, %537, %264) : (tensor<f32>, tensor<1x14x14x96xf32>, tensor<f32>) -> tensor<1x14x14x96xf32>
    %539 = mhlo.multiply %538, %232 : tensor<1x14x14x96xf32>
    %540 = mhlo.multiply %539, %536 : tensor<1x14x14x96xf32>
    %541 = "mhlo.reduce"(%540, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<f32>) -> tensor<1x96xf32>
    %542 = mhlo.divide %541, %252 : tensor<1x96xf32>
    %543 = "mhlo.reshape"(%542) : (tensor<1x96xf32>) -> tensor<1x1x1x96xf32>
    %544 = "mhlo.convolution"(%543, %346) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x96xf32>, tensor<1x1x96x24xf32>) -> tensor<1x1x1x24xf32>
    %545 = "mhlo.broadcast_in_dim"(%345) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %546 = mhlo.add %544, %545 : tensor<1x1x1x24xf32>
    %547 = mhlo.maximum %546, %253 : tensor<1x1x1x24xf32>
    %548 = "mhlo.convolution"(%547, %344) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x24xf32>, tensor<1x1x24x96xf32>) -> tensor<1x1x1x96xf32>
    %549 = "mhlo.broadcast_in_dim"(%343) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %550 = mhlo.add %548, %549 : tensor<1x1x1x96xf32>
    %551 = mhlo.add %550, %216 : tensor<1x1x1x96xf32>
    %552 = "mhlo.clamp"(%266, %551, %264) : (tensor<f32>, tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    %553 = mhlo.multiply %552, %233 : tensor<1x1x1x96xf32>
    %554 = "mhlo.broadcast_in_dim"(%553) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %555 = mhlo.multiply %540, %554 : tensor<1x14x14x96xf32>
    %556 = "mhlo.convolution"(%555, %342) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<1x1x96x40xf32>) -> tensor<1x14x14x40xf32>
    %557 = "mhlo.batch_norm_inference"(%556, %341, %340, %339, %338) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %558 = "mhlo.convolution"(%557, %356) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) -> tensor<1x14x14x240xf32>
    %559 = "mhlo.batch_norm_inference"(%558, %355, %354, %353, %352) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %560 = mhlo.add %559, %217 : tensor<1x14x14x240xf32>
    %561 = "mhlo.clamp"(%266, %560, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %562 = mhlo.multiply %561, %234 : tensor<1x14x14x240xf32>
    %563 = mhlo.multiply %562, %559 : tensor<1x14x14x240xf32>
    %564 = "mhlo.reshape"(%351) : (tensor<5x5x240x1xf32>) -> tensor<5x5x1x240xf32>
    %565 = "mhlo.convolution"(%563, %564) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 240 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<5x5x1x240xf32>) -> tensor<1x14x14x240xf32>
    %566 = "mhlo.batch_norm_inference"(%565, %350, %349, %348, %347) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %567 = mhlo.add %566, %217 : tensor<1x14x14x240xf32>
    %568 = "mhlo.clamp"(%266, %567, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %569 = mhlo.multiply %568, %234 : tensor<1x14x14x240xf32>
    %570 = mhlo.multiply %569, %566 : tensor<1x14x14x240xf32>
    %571 = "mhlo.reduce"(%570, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x240xf32>
    %572 = mhlo.divide %571, %254 : tensor<1x240xf32>
    %573 = "mhlo.reshape"(%572) : (tensor<1x240xf32>) -> tensor<1x1x1x240xf32>
    %574 = "mhlo.convolution"(%573, %365) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) -> tensor<1x1x1x64xf32>
    %575 = "mhlo.broadcast_in_dim"(%364) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %576 = mhlo.add %574, %575 : tensor<1x1x1x64xf32>
    %577 = mhlo.maximum %576, %255 : tensor<1x1x1x64xf32>
    %578 = "mhlo.convolution"(%577, %363) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) -> tensor<1x1x1x240xf32>
    %579 = "mhlo.broadcast_in_dim"(%362) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %580 = mhlo.add %578, %579 : tensor<1x1x1x240xf32>
    %581 = mhlo.add %580, %218 : tensor<1x1x1x240xf32>
    %582 = "mhlo.clamp"(%266, %581, %264) : (tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) -> tensor<1x1x1x240xf32>
    %583 = mhlo.multiply %582, %235 : tensor<1x1x1x240xf32>
    %584 = "mhlo.broadcast_in_dim"(%583) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x240xf32>) -> tensor<1x14x14x240xf32>
    %585 = mhlo.multiply %570, %584 : tensor<1x14x14x240xf32>
    %586 = "mhlo.convolution"(%585, %361) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) -> tensor<1x14x14x40xf32>
    %587 = "mhlo.batch_norm_inference"(%586, %360, %359, %358, %357) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %588 = mhlo.add %557, %587 : tensor<1x14x14x40xf32>
    %589 = "mhlo.convolution"(%588, %375) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) -> tensor<1x14x14x240xf32>
    %590 = "mhlo.batch_norm_inference"(%589, %374, %373, %372, %371) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %591 = mhlo.add %590, %217 : tensor<1x14x14x240xf32>
    %592 = "mhlo.clamp"(%266, %591, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %593 = mhlo.multiply %592, %234 : tensor<1x14x14x240xf32>
    %594 = mhlo.multiply %593, %590 : tensor<1x14x14x240xf32>
    %595 = "mhlo.reshape"(%370) : (tensor<5x5x240x1xf32>) -> tensor<5x5x1x240xf32>
    %596 = "mhlo.convolution"(%594, %595) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 240 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<5x5x1x240xf32>) -> tensor<1x14x14x240xf32>
    %597 = "mhlo.batch_norm_inference"(%596, %369, %368, %367, %366) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %598 = mhlo.add %597, %217 : tensor<1x14x14x240xf32>
    %599 = "mhlo.clamp"(%266, %598, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %600 = mhlo.multiply %599, %234 : tensor<1x14x14x240xf32>
    %601 = mhlo.multiply %600, %597 : tensor<1x14x14x240xf32>
    %602 = "mhlo.reduce"(%601, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x240xf32>
    %603 = mhlo.divide %602, %254 : tensor<1x240xf32>
    %604 = "mhlo.reshape"(%603) : (tensor<1x240xf32>) -> tensor<1x1x1x240xf32>
    %605 = "mhlo.convolution"(%604, %384) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) -> tensor<1x1x1x64xf32>
    %606 = "mhlo.broadcast_in_dim"(%383) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %607 = mhlo.add %605, %606 : tensor<1x1x1x64xf32>
    %608 = mhlo.maximum %607, %255 : tensor<1x1x1x64xf32>
    %609 = "mhlo.convolution"(%608, %382) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) -> tensor<1x1x1x240xf32>
    %610 = "mhlo.broadcast_in_dim"(%381) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %611 = mhlo.add %609, %610 : tensor<1x1x1x240xf32>
    %612 = mhlo.add %611, %218 : tensor<1x1x1x240xf32>
    %613 = "mhlo.clamp"(%266, %612, %264) : (tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) -> tensor<1x1x1x240xf32>
    %614 = mhlo.multiply %613, %235 : tensor<1x1x1x240xf32>
    %615 = "mhlo.broadcast_in_dim"(%614) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x240xf32>) -> tensor<1x14x14x240xf32>
    %616 = mhlo.multiply %601, %615 : tensor<1x14x14x240xf32>
    %617 = "mhlo.convolution"(%616, %380) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) -> tensor<1x14x14x40xf32>
    %618 = "mhlo.batch_norm_inference"(%617, %379, %378, %377, %376) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %619 = mhlo.add %588, %618 : tensor<1x14x14x40xf32>
    %620 = "mhlo.convolution"(%619, %394) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x120xf32>) -> tensor<1x14x14x120xf32>
    %621 = "mhlo.batch_norm_inference"(%620, %393, %392, %391, %390) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %622 = mhlo.add %621, %219 : tensor<1x14x14x120xf32>
    %623 = "mhlo.clamp"(%266, %622, %264) : (tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x14x14x120xf32>
    %624 = mhlo.multiply %623, %236 : tensor<1x14x14x120xf32>
    %625 = mhlo.multiply %624, %621 : tensor<1x14x14x120xf32>
    %626 = "mhlo.reshape"(%389) : (tensor<5x5x120x1xf32>) -> tensor<5x5x1x120xf32>
    %627 = "mhlo.convolution"(%625, %626) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 120 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<5x5x1x120xf32>) -> tensor<1x14x14x120xf32>
    %628 = "mhlo.batch_norm_inference"(%627, %388, %387, %386, %385) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %629 = mhlo.add %628, %219 : tensor<1x14x14x120xf32>
    %630 = "mhlo.clamp"(%266, %629, %264) : (tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x14x14x120xf32>
    %631 = mhlo.multiply %630, %236 : tensor<1x14x14x120xf32>
    %632 = mhlo.multiply %631, %628 : tensor<1x14x14x120xf32>
    %633 = "mhlo.reduce"(%632, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x120xf32>
    %634 = mhlo.divide %633, %256 : tensor<1x120xf32>
    %635 = "mhlo.reshape"(%634) : (tensor<1x120xf32>) -> tensor<1x1x1x120xf32>
    %636 = "mhlo.convolution"(%635, %403) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x120xf32>, tensor<1x1x120x32xf32>) -> tensor<1x1x1x32xf32>
    %637 = "mhlo.broadcast_in_dim"(%402) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %638 = mhlo.add %636, %637 : tensor<1x1x1x32xf32>
    %639 = mhlo.maximum %638, %257 : tensor<1x1x1x32xf32>
    %640 = "mhlo.convolution"(%639, %401) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x32xf32>, tensor<1x1x32x120xf32>) -> tensor<1x1x1x120xf32>
    %641 = "mhlo.broadcast_in_dim"(%400) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %642 = mhlo.add %640, %641 : tensor<1x1x1x120xf32>
    %643 = mhlo.add %642, %220 : tensor<1x1x1x120xf32>
    %644 = "mhlo.clamp"(%266, %643, %264) : (tensor<f32>, tensor<1x1x1x120xf32>, tensor<f32>) -> tensor<1x1x1x120xf32>
    %645 = mhlo.multiply %644, %237 : tensor<1x1x1x120xf32>
    %646 = "mhlo.broadcast_in_dim"(%645) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x120xf32>) -> tensor<1x14x14x120xf32>
    %647 = mhlo.multiply %632, %646 : tensor<1x14x14x120xf32>
    %648 = "mhlo.convolution"(%647, %399) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<1x1x120x48xf32>) -> tensor<1x14x14x48xf32>
    %649 = "mhlo.batch_norm_inference"(%648, %398, %397, %396, %395) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %650 = "mhlo.convolution"(%649, %413) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x48xf32>, tensor<1x1x48x144xf32>) -> tensor<1x14x14x144xf32>
    %651 = "mhlo.batch_norm_inference"(%650, %412, %411, %410, %409) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %652 = mhlo.add %651, %221 : tensor<1x14x14x144xf32>
    %653 = "mhlo.clamp"(%266, %652, %264) : (tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x14x14x144xf32>
    %654 = mhlo.multiply %653, %238 : tensor<1x14x14x144xf32>
    %655 = mhlo.multiply %654, %651 : tensor<1x14x14x144xf32>
    %656 = "mhlo.reshape"(%408) : (tensor<5x5x144x1xf32>) -> tensor<5x5x1x144xf32>
    %657 = "mhlo.convolution"(%655, %656) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 144 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<5x5x1x144xf32>) -> tensor<1x14x14x144xf32>
    %658 = "mhlo.batch_norm_inference"(%657, %407, %406, %405, %404) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %659 = mhlo.add %658, %221 : tensor<1x14x14x144xf32>
    %660 = "mhlo.clamp"(%266, %659, %264) : (tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x14x14x144xf32>
    %661 = mhlo.multiply %660, %238 : tensor<1x14x14x144xf32>
    %662 = mhlo.multiply %661, %658 : tensor<1x14x14x144xf32>
    %663 = "mhlo.reduce"(%662, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x144xf32>
    %664 = mhlo.divide %663, %258 : tensor<1x144xf32>
    %665 = "mhlo.reshape"(%664) : (tensor<1x144xf32>) -> tensor<1x1x1x144xf32>
    %666 = "mhlo.convolution"(%665, %422) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x40xf32>) -> tensor<1x1x1x40xf32>
    %667 = "mhlo.broadcast_in_dim"(%421) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<40xf32>) -> tensor<1x1x1x40xf32>
    %668 = mhlo.add %666, %667 : tensor<1x1x1x40xf32>
    %669 = mhlo.maximum %668, %259 : tensor<1x1x1x40xf32>
    %670 = "mhlo.convolution"(%669, %420) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x40xf32>, tensor<1x1x40x144xf32>) -> tensor<1x1x1x144xf32>
    %671 = "mhlo.broadcast_in_dim"(%419) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %672 = mhlo.add %670, %671 : tensor<1x1x1x144xf32>
    %673 = mhlo.add %672, %222 : tensor<1x1x1x144xf32>
    %674 = "mhlo.clamp"(%266, %673, %264) : (tensor<f32>, tensor<1x1x1x144xf32>, tensor<f32>) -> tensor<1x1x1x144xf32>
    %675 = mhlo.multiply %674, %239 : tensor<1x1x1x144xf32>
    %676 = "mhlo.broadcast_in_dim"(%675) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x144xf32>) -> tensor<1x14x14x144xf32>
    %677 = mhlo.multiply %662, %676 : tensor<1x14x14x144xf32>
    %678 = "mhlo.convolution"(%677, %418) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<1x1x144x48xf32>) -> tensor<1x14x14x48xf32>
    %679 = "mhlo.batch_norm_inference"(%678, %417, %416, %415, %414) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %680 = mhlo.add %649, %679 : tensor<1x14x14x48xf32>
    %681 = "mhlo.convolution"(%680, %432) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x48xf32>, tensor<1x1x48x288xf32>) -> tensor<1x14x14x288xf32>
    %682 = "mhlo.batch_norm_inference"(%681, %431, %430, %429, %428) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> tensor<1x14x14x288xf32>
    %683 = mhlo.add %682, %223 : tensor<1x14x14x288xf32>
    %684 = "mhlo.clamp"(%266, %683, %264) : (tensor<f32>, tensor<1x14x14x288xf32>, tensor<f32>) -> tensor<1x14x14x288xf32>
    %685 = mhlo.multiply %684, %240 : tensor<1x14x14x288xf32>
    %686 = mhlo.multiply %685, %682 : tensor<1x14x14x288xf32>
    %687 = "mhlo.pad"(%686, %266) {edge_padding_high = dense<[0, 2, 2, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x14x14x288xf32>, tensor<f32>) -> tensor<1x17x17x288xf32>
    %688 = "mhlo.reshape"(%427) : (tensor<5x5x288x1xf32>) -> tensor<5x5x1x288xf32>
    %689 = "mhlo.convolution"(%687, %688) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 288 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x17x17x288xf32>, tensor<5x5x1x288xf32>) -> tensor<1x7x7x288xf32>
    %690 = "mhlo.batch_norm_inference"(%689, %426, %425, %424, %423) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> tensor<1x7x7x288xf32>
    %691 = mhlo.add %690, %224 : tensor<1x7x7x288xf32>
    %692 = "mhlo.clamp"(%266, %691, %264) : (tensor<f32>, tensor<1x7x7x288xf32>, tensor<f32>) -> tensor<1x7x7x288xf32>
    %693 = mhlo.multiply %692, %241 : tensor<1x7x7x288xf32>
    %694 = mhlo.multiply %693, %690 : tensor<1x7x7x288xf32>
    %695 = "mhlo.reduce"(%694, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x288xf32>, tensor<f32>) -> tensor<1x288xf32>
    %696 = mhlo.divide %695, %260 : tensor<1x288xf32>
    %697 = "mhlo.reshape"(%696) : (tensor<1x288xf32>) -> tensor<1x1x1x288xf32>
    %698 = "mhlo.convolution"(%697, %441) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x288xf32>, tensor<1x1x288x72xf32>) -> tensor<1x1x1x72xf32>
    %699 = "mhlo.broadcast_in_dim"(%440) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<72xf32>) -> tensor<1x1x1x72xf32>
    %700 = mhlo.add %698, %699 : tensor<1x1x1x72xf32>
    %701 = mhlo.maximum %700, %261 : tensor<1x1x1x72xf32>
    %702 = "mhlo.convolution"(%701, %439) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x72xf32>, tensor<1x1x72x288xf32>) -> tensor<1x1x1x288xf32>
    %703 = "mhlo.broadcast_in_dim"(%438) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<288xf32>) -> tensor<1x1x1x288xf32>
    %704 = mhlo.add %702, %703 : tensor<1x1x1x288xf32>
    %705 = mhlo.add %704, %225 : tensor<1x1x1x288xf32>
    %706 = "mhlo.clamp"(%266, %705, %264) : (tensor<f32>, tensor<1x1x1x288xf32>, tensor<f32>) -> tensor<1x1x1x288xf32>
    %707 = mhlo.multiply %706, %242 : tensor<1x1x1x288xf32>
    %708 = "mhlo.broadcast_in_dim"(%707) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x288xf32>) -> tensor<1x7x7x288xf32>
    %709 = mhlo.multiply %694, %708 : tensor<1x7x7x288xf32>
    %710 = "mhlo.convolution"(%709, %437) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x288xf32>, tensor<1x1x288x96xf32>) -> tensor<1x7x7x96xf32>
    %711 = "mhlo.batch_norm_inference"(%710, %436, %435, %434, %433) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %712 = "mhlo.convolution"(%711, %451) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %713 = "mhlo.batch_norm_inference"(%712, %450, %449, %448, %447) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %714 = mhlo.add %713, %227 : tensor<1x7x7x576xf32>
    %715 = "mhlo.clamp"(%266, %714, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %716 = mhlo.multiply %715, %244 : tensor<1x7x7x576xf32>
    %717 = mhlo.multiply %716, %713 : tensor<1x7x7x576xf32>
    %718 = "mhlo.reshape"(%446) : (tensor<5x5x576x1xf32>) -> tensor<5x5x1x576xf32>
    %719 = "mhlo.convolution"(%717, %718) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 576 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<5x5x1x576xf32>) -> tensor<1x7x7x576xf32>
    %720 = "mhlo.batch_norm_inference"(%719, %445, %444, %443, %442) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %721 = mhlo.add %720, %227 : tensor<1x7x7x576xf32>
    %722 = "mhlo.clamp"(%266, %721, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %723 = mhlo.multiply %722, %244 : tensor<1x7x7x576xf32>
    %724 = mhlo.multiply %723, %720 : tensor<1x7x7x576xf32>
    %725 = "mhlo.reduce"(%724, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %726 = mhlo.divide %725, %263 : tensor<1x576xf32>
    %727 = "mhlo.reshape"(%726) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %728 = "mhlo.convolution"(%727, %460) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) -> tensor<1x1x1x144xf32>
    %729 = "mhlo.broadcast_in_dim"(%459) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %730 = mhlo.add %728, %729 : tensor<1x1x1x144xf32>
    %731 = mhlo.maximum %730, %262 : tensor<1x1x1x144xf32>
    %732 = "mhlo.convolution"(%731, %458) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) -> tensor<1x1x1x576xf32>
    %733 = "mhlo.broadcast_in_dim"(%457) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %734 = mhlo.add %732, %733 : tensor<1x1x1x576xf32>
    %735 = mhlo.add %734, %226 : tensor<1x1x1x576xf32>
    %736 = "mhlo.clamp"(%266, %735, %264) : (tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) -> tensor<1x1x1x576xf32>
    %737 = mhlo.multiply %736, %243 : tensor<1x1x1x576xf32>
    %738 = "mhlo.broadcast_in_dim"(%737) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %739 = mhlo.multiply %724, %738 : tensor<1x7x7x576xf32>
    %740 = "mhlo.convolution"(%739, %456) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x7x7x96xf32>
    %741 = "mhlo.batch_norm_inference"(%740, %455, %454, %453, %452) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %742 = mhlo.add %711, %741 : tensor<1x7x7x96xf32>
    %743 = "mhlo.convolution"(%742, %288) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %744 = "mhlo.batch_norm_inference"(%743, %287, %286, %285, %284) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %745 = mhlo.add %744, %227 : tensor<1x7x7x576xf32>
    %746 = "mhlo.clamp"(%266, %745, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %747 = mhlo.multiply %746, %244 : tensor<1x7x7x576xf32>
    %748 = mhlo.multiply %747, %744 : tensor<1x7x7x576xf32>
    %749 = "mhlo.reshape"(%283) : (tensor<5x5x576x1xf32>) -> tensor<5x5x1x576xf32>
    %750 = "mhlo.convolution"(%748, %749) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 576 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<5x5x1x576xf32>) -> tensor<1x7x7x576xf32>
    %751 = "mhlo.batch_norm_inference"(%750, %282, %281, %280, %279) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %752 = mhlo.add %751, %227 : tensor<1x7x7x576xf32>
    %753 = "mhlo.clamp"(%266, %752, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %754 = mhlo.multiply %753, %244 : tensor<1x7x7x576xf32>
    %755 = mhlo.multiply %754, %751 : tensor<1x7x7x576xf32>
    %756 = "mhlo.reduce"(%755, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %757 = mhlo.divide %756, %263 : tensor<1x576xf32>
    %758 = "mhlo.reshape"(%757) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %759 = "mhlo.convolution"(%758, %297) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) -> tensor<1x1x1x144xf32>
    %760 = "mhlo.broadcast_in_dim"(%296) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %761 = mhlo.add %759, %760 : tensor<1x1x1x144xf32>
    %762 = mhlo.maximum %761, %262 : tensor<1x1x1x144xf32>
    %763 = "mhlo.convolution"(%762, %295) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) -> tensor<1x1x1x576xf32>
    %764 = "mhlo.broadcast_in_dim"(%294) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %765 = mhlo.add %763, %764 : tensor<1x1x1x576xf32>
    %766 = mhlo.add %765, %226 : tensor<1x1x1x576xf32>
    %767 = "mhlo.clamp"(%266, %766, %264) : (tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) -> tensor<1x1x1x576xf32>
    %768 = mhlo.multiply %767, %243 : tensor<1x1x1x576xf32>
    %769 = "mhlo.broadcast_in_dim"(%768) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %770 = mhlo.multiply %755, %769 : tensor<1x7x7x576xf32>
    %771 = "mhlo.convolution"(%770, %293) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x7x7x96xf32>
    %772 = "mhlo.batch_norm_inference"(%771, %292, %291, %290, %289) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %773 = mhlo.add %742, %772 : tensor<1x7x7x96xf32>
    %774 = "mhlo.convolution"(%773, %271) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %775 = "mhlo.batch_norm_inference"(%774, %270, %269, %268, %267) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %776 = mhlo.add %775, %227 : tensor<1x7x7x576xf32>
    %777 = "mhlo.clamp"(%266, %776, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %778 = mhlo.multiply %777, %244 : tensor<1x7x7x576xf32>
    %779 = mhlo.multiply %778, %775 : tensor<1x7x7x576xf32>
    %780 = "mhlo.reduce"(%779, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %781 = mhlo.divide %780, %263 : tensor<1x576xf32>
    %782 = "mhlo.reshape"(%781) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %783 = "mhlo.convolution"(%782, %273) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x1024xf32>) -> tensor<1x1x1x1024xf32>
    %784 = "mhlo.broadcast_in_dim"(%272) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %785 = mhlo.add %783, %784 : tensor<1x1x1x1024xf32>
    %786 = mhlo.add %785, %228 : tensor<1x1x1x1024xf32>
    %787 = "mhlo.clamp"(%266, %786, %264) : (tensor<f32>, tensor<1x1x1x1024xf32>, tensor<f32>) -> tensor<1x1x1x1024xf32>
    %788 = mhlo.multiply %787, %245 : tensor<1x1x1x1024xf32>
    %789 = mhlo.multiply %788, %785 : tensor<1x1x1x1024xf32>
    %790 = "mhlo.convolution"(%789, %476) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x1024xf32>, tensor<1x1x1024x1000xf32>) -> tensor<1x1x1x1000xf32>
    %791 = "mhlo.broadcast_in_dim"(%475) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1x1x1000xf32>
    %792 = mhlo.add %790, %791 : tensor<1x1x1x1000xf32>
    %793 = "mhlo.reshape"(%792) : (tensor<1x1x1x1000xf32>) -> tensor<1x1000xf32>
    %794 = "mhlo.reduce"(%793, %265) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %795 = "mhlo.broadcast_in_dim"(%794) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %796 = mhlo.subtract %793, %795 : tensor<1x1000xf32>
    %797 = "mhlo.exponential"(%796) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %798 = "mhlo.reduce"(%797, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %799 = "mhlo.broadcast_in_dim"(%798) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %800 = mhlo.divide %797, %799 : tensor<1x1000xf32>
    check.expect_almost_eq_const(%800, dense<0.001> : tensor<1x1000xf32>) : tensor<1x1000xf32>
    return
  }
}


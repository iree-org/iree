// MobileNet v2 model with placeholder weights, for testing and development use.

module  {
  flow.variable @"__iree_flow___sm_node163__m.layer-1.kernel" dense<1.000000e+00> : tensor<3x3x3x32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node169__m.layer-2.gamma" dense<5.000000e-01> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node170__m.layer-2.beta" dense<0.333333343> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node171__m.layer-2.moving_mean" dense<2.500000e-01> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node172__m.layer-2.moving_variance" dense<2.000000e-01> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node181__m.layer-4.depthwise_kernel" dense<0.166666672> : tensor<3x3x32x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node187__m.layer-5.gamma" dense<0.142857149> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node188__m.layer-5.beta" dense<1.250000e-01> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node189__m.layer-5.moving_mean" dense<0.111111112> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node190__m.layer-5.moving_variance" dense<1.000000e-01> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node199__m.layer-7.kernel" dense<0.0909090936> : tensor<1x1x32x16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node205__m.layer-8.gamma" dense<0.0833333358> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node206__m.layer-8.beta" dense<0.0769230798> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node207__m.layer-8.moving_mean" dense<0.0714285746> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node208__m.layer-8.moving_variance" dense<0.0666666701> : tensor<16xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node213__m.layer-9.kernel" dense<6.250000e-02> : tensor<1x1x16x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node219__m.layer-10.gamma" dense<0.0588235296> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node220__m.layer-10.beta" dense<0.055555556> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node221__m.layer-10.moving_mean" dense<0.0526315793> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node222__m.layer-10.moving_variance" dense<5.000000e-02> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node235__m.layer-13.depthwise_kernel" dense<0.0476190485> : tensor<3x3x96x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node241__m.layer-14.gamma" dense<0.0454545468> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node242__m.layer-14.beta" dense<0.0434782617> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node243__m.layer-14.moving_mean" dense<0.0416666679> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node244__m.layer-14.moving_variance" dense<4.000000e-02> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node253__m.layer-16.kernel" dense<0.0384615399> : tensor<1x1x96x24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node259__m.layer-17.gamma" dense<0.0370370373> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node260__m.layer-17.beta" dense<0.0357142873> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node261__m.layer-17.moving_mean" dense<0.0344827585> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node262__m.layer-17.moving_variance" dense<0.0333333351> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node267__m.layer-18.kernel" dense<0.0322580636> : tensor<1x1x24x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node273__m.layer-19.gamma" dense<3.125000e-02> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node274__m.layer-19.beta" dense<0.0303030312> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node275__m.layer-19.moving_mean" dense<0.0294117648> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node276__m.layer-19.moving_variance" dense<0.0285714287> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node285__m.layer-21.depthwise_kernel" dense<0.027777778> : tensor<3x3x144x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node291__m.layer-22.gamma" dense<0.0270270277> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node292__m.layer-22.beta" dense<0.0263157897> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node293__m.layer-22.moving_mean" dense<0.025641026> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node294__m.layer-22.moving_variance" dense<2.500000e-02> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node303__m.layer-24.kernel" dense<0.024390243> : tensor<1x1x144x24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node309__m.layer-25.gamma" dense<0.0238095243> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node310__m.layer-25.beta" dense<0.0232558139> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node311__m.layer-25.moving_mean" dense<0.0227272734> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node312__m.layer-25.moving_variance" dense<0.0222222228> : tensor<24xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node321__m.layer-27.kernel" dense<0.0217391308> : tensor<1x1x24x144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node327__m.layer-28.gamma" dense<0.0212765951> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node328__m.layer-28.beta" dense<0.020833334> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node329__m.layer-28.moving_mean" dense<0.0204081628> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node330__m.layer-28.moving_variance" dense<2.000000e-02> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node343__m.layer-31.depthwise_kernel" dense<0.0196078438> : tensor<3x3x144x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node349__m.layer-32.gamma" dense<0.0192307699> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node350__m.layer-32.beta" dense<0.0188679248> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node351__m.layer-32.moving_mean" dense<0.0185185187> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node352__m.layer-32.moving_variance" dense<0.0181818176> : tensor<144xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node361__m.layer-34.kernel" dense<0.0178571437> : tensor<1x1x144x32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node367__m.layer-35.gamma" dense<0.0175438598> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node368__m.layer-35.beta" dense<0.0172413792> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node369__m.layer-35.moving_mean" dense<0.0169491526> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node370__m.layer-35.moving_variance" dense<0.0166666675> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node375__m.layer-36.kernel" dense<0.0163934417> : tensor<1x1x32x192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node381__m.layer-37.gamma" dense<0.0161290318> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node382__m.layer-37.beta" dense<0.0158730168> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node383__m.layer-37.moving_mean" dense<1.562500e-02> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node384__m.layer-37.moving_variance" dense<0.0153846154> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node393__m.layer-39.depthwise_kernel" dense<0.0151515156> : tensor<3x3x192x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node399__m.layer-40.gamma" dense<0.0149253728> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node400__m.layer-40.beta" dense<0.0147058824> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node401__m.layer-40.moving_mean" dense<0.0144927539> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node402__m.layer-40.moving_variance" dense<0.0142857144> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node411__m.layer-42.kernel" dense<0.0140845068> : tensor<1x1x192x32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node417__m.layer-43.gamma" dense<0.013888889> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node418__m.layer-43.beta" dense<0.01369863> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node419__m.layer-43.moving_mean" dense<0.0135135138> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node420__m.layer-43.moving_variance" dense<0.0133333337> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node429__m.layer-45.kernel" dense<0.0131578948> : tensor<1x1x32x192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node435__m.layer-46.gamma" dense<0.012987013> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node436__m.layer-46.beta" dense<0.012820513> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node437__m.layer-46.moving_mean" dense<0.0126582282> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node438__m.layer-46.moving_variance" dense<1.250000e-02> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node447__m.layer-48.depthwise_kernel" dense<0.0123456791> : tensor<3x3x192x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node453__m.layer-49.gamma" dense<0.0121951215> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node454__m.layer-49.beta" dense<0.0120481923> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node455__m.layer-49.moving_mean" dense<0.0119047621> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node456__m.layer-49.moving_variance" dense<0.0117647061> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node465__m.layer-51.kernel" dense<0.0116279069> : tensor<1x1x192x32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node471__m.layer-52.gamma" dense<0.0114942528> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node472__m.layer-52.beta" dense<0.0113636367> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node473__m.layer-52.moving_mean" dense<0.0112359552> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node474__m.layer-52.moving_variance" dense<0.0111111114> : tensor<32xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node483__m.layer-54.kernel" dense<0.0109890113> : tensor<1x1x32x192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node489__m.layer-55.gamma" dense<0.0108695654> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node490__m.layer-55.beta" dense<0.0107526882> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node491__m.layer-55.moving_mean" dense<0.0106382975> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node492__m.layer-55.moving_variance" dense<0.0105263162> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node505__m.layer-58.depthwise_kernel" dense<0.010416667> : tensor<3x3x192x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node511__m.layer-59.gamma" dense<0.010309278> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node512__m.layer-59.beta" dense<0.0102040814> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node513__m.layer-59.moving_mean" dense<0.0101010101> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node514__m.layer-59.moving_variance" dense<0.00999999977> : tensor<192xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node523__m.layer-61.kernel" dense<9.900990e-03> : tensor<1x1x192x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node529__m.layer-62.gamma" dense<0.00980392192> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node530__m.layer-62.beta" dense<0.00970873795> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node531__m.layer-62.moving_mean" dense<0.00961538497> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node532__m.layer-62.moving_variance" dense<9.523810e-03> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node537__m.layer-63.kernel" dense<0.0094339624> : tensor<1x1x64x384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node543__m.layer-64.gamma" dense<0.00934579409> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node544__m.layer-64.beta" dense<0.00925925932> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node545__m.layer-64.moving_mean" dense<0.00917431153> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node546__m.layer-64.moving_variance" dense<0.0090909088> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node555__m.layer-66.depthwise_kernel" dense<0.00900900922> : tensor<3x3x384x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node561__m.layer-67.gamma" dense<0.00892857183> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node562__m.layer-67.beta" dense<0.00884955748> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node563__m.layer-67.moving_mean" dense<0.00877192988> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node564__m.layer-67.moving_variance" dense<0.00869565178> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node573__m.layer-69.kernel" dense<8.620690e-03> : tensor<1x1x384x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node579__m.layer-70.gamma" dense<0.00854700897> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node580__m.layer-70.beta" dense<0.00847457629> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node581__m.layer-70.moving_mean" dense<0.00840336177> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node582__m.layer-70.moving_variance" dense<0.00833333377> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node591__m.layer-72.kernel" dense<0.00826446246> : tensor<1x1x64x384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node597__m.layer-73.gamma" dense<0.00819672085> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node598__m.layer-73.beta" dense<0.008130081> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node599__m.layer-73.moving_mean" dense<0.00806451589> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node600__m.layer-73.moving_variance" dense<8.000000e-03> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node609__m.layer-75.depthwise_kernel" dense<0.00793650839> : tensor<3x3x384x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node615__m.layer-76.gamma" dense<0.00787401571> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node616__m.layer-76.beta" dense<7.812500e-03> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node617__m.layer-76.moving_mean" dense<0.00775193795> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node618__m.layer-76.moving_variance" dense<0.0076923077> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node627__m.layer-78.kernel" dense<0.00763358781> : tensor<1x1x384x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node633__m.layer-79.gamma" dense<0.0075757578> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node634__m.layer-79.beta" dense<0.00751879718> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node635__m.layer-79.moving_mean" dense<0.00746268639> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node636__m.layer-79.moving_variance" dense<0.00740740728> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node645__m.layer-81.kernel" dense<0.0073529412> : tensor<1x1x64x384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node651__m.layer-82.gamma" dense<7.299270e-03> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node652__m.layer-82.beta" dense<0.00724637694> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node653__m.layer-82.moving_mean" dense<0.00719424477> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node654__m.layer-82.moving_variance" dense<0.00714285718> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node663__m.layer-84.depthwise_kernel" dense<0.00709219835> : tensor<3x3x384x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node669__m.layer-85.gamma" dense<0.00704225338> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node670__m.layer-85.beta" dense<0.00699300691> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node671__m.layer-85.moving_mean" dense<0.0069444445> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node672__m.layer-85.moving_variance" dense<0.0068965517> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node681__m.layer-87.kernel" dense<0.00684931502> : tensor<1x1x384x64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node687__m.layer-88.gamma" dense<0.00680272094> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node688__m.layer-88.beta" dense<0.00675675692> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node689__m.layer-88.moving_mean" dense<0.00671140943> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node690__m.layer-88.moving_variance" dense<0.00666666683> : tensor<64xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node699__m.layer-90.kernel" dense<0.00662251655> : tensor<1x1x64x384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node705__m.layer-91.gamma" dense<0.00657894742> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node706__m.layer-91.beta" dense<0.00653594779> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node707__m.layer-91.moving_mean" dense<0.00649350649> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node708__m.layer-91.moving_variance" dense<0.0064516128> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node717__m.layer-93.depthwise_kernel" dense<0.00641025649> : tensor<3x3x384x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node723__m.layer-94.gamma" dense<0.00636942684> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node724__m.layer-94.beta" dense<0.00632911408> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node725__m.layer-94.moving_mean" dense<0.00628930796> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node726__m.layer-94.moving_variance" dense<6.250000e-03> : tensor<384xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node735__m.layer-96.kernel" dense<0.00621118024> : tensor<1x1x384x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node741__m.layer-97.gamma" dense<0.00617283955> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node742__m.layer-97.beta" dense<0.00613496918> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node743__m.layer-97.moving_mean" dense<0.00609756075> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node744__m.layer-97.moving_variance" dense<0.00606060587> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node749__m.layer-98.kernel" dense<0.00602409616> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node755__m.layer-99.gamma" dense<0.00598802418> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node756__m.layer-99.beta" dense<0.00595238106> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node757__m.layer-99.moving_mean" dense<5.917160e-03> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node758__m.layer-99.moving_variance" dense<0.00588235306> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node767__m.layer-101.depthwise_kernel" dense<0.00584795326> : tensor<3x3x576x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node773__m.layer-102.gamma" dense<0.00581395347> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node774__m.layer-102.beta" dense<0.00578034669> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node775__m.layer-102.moving_mean" dense<0.00574712642> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node776__m.layer-102.moving_variance" dense<0.00571428565> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node785__m.layer-104.kernel" dense<0.00568181835> : tensor<1x1x576x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node791__m.layer-105.gamma" dense<0.00564971752> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node792__m.layer-105.beta" dense<0.00561797759> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node793__m.layer-105.moving_mean" dense<0.00558659201> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node794__m.layer-105.moving_variance" dense<0.00555555569> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node803__m.layer-107.kernel" dense<0.00552486209> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node809__m.layer-108.gamma" dense<0.00549450563> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node810__m.layer-108.beta" dense<0.00546448072> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node811__m.layer-108.moving_mean" dense<0.00543478271> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node812__m.layer-108.moving_variance" dense<0.00540540554> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node821__m.layer-110.depthwise_kernel" dense<0.00537634408> : tensor<3x3x576x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node827__m.layer-111.gamma" dense<0.00534759369> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node828__m.layer-111.beta" dense<0.00531914877> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node829__m.layer-111.moving_mean" dense<0.00529100513> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node830__m.layer-111.moving_variance" dense<0.00526315812> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node839__m.layer-113.kernel" dense<0.00523560215> : tensor<1x1x576x96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node845__m.layer-114.gamma" dense<0.00520833349> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node846__m.layer-114.beta" dense<0.00518134702> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node847__m.layer-114.moving_mean" dense<0.00515463902> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node848__m.layer-114.moving_variance" dense<0.00512820529> : tensor<96xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node857__m.layer-116.kernel" dense<0.00510204071> : tensor<1x1x96x576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node863__m.layer-117.gamma" dense<0.00507614203> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node864__m.layer-117.beta" dense<0.00505050505> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node865__m.layer-117.moving_mean" dense<0.00502512557> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node866__m.layer-117.moving_variance" dense<5.000000e-03> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node879__m.layer-120.depthwise_kernel" dense<0.00497512426> : tensor<3x3x576x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node885__m.layer-121.gamma" dense<0.00495049497> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node886__m.layer-121.beta" dense<0.00492610829> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node887__m.layer-121.moving_mean" dense<0.00490196096> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node888__m.layer-121.moving_variance" dense<0.00487804879> : tensor<576xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node897__m.layer-123.kernel" dense<0.00485436898> : tensor<1x1x576x160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node903__m.layer-124.gamma" dense<0.00483091781> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node904__m.layer-124.beta" dense<0.00480769249> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node905__m.layer-124.moving_mean" dense<0.00478468882> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node906__m.layer-124.moving_variance" dense<0.00476190494> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node911__m.layer-125.kernel" dense<0.00473933667> : tensor<1x1x160x960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node917__m.layer-126.gamma" dense<0.0047169812> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node918__m.layer-126.beta" dense<0.00469483575> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node919__m.layer-126.moving_mean" dense<0.00467289705> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node920__m.layer-126.moving_variance" dense<0.00465116277> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node929__m.layer-128.depthwise_kernel" dense<0.00462962966> : tensor<3x3x960x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node935__m.layer-129.gamma" dense<0.00460829493> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node936__m.layer-129.beta" dense<0.00458715577> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node937__m.layer-129.moving_mean" dense<4.566210e-03> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node938__m.layer-129.moving_variance" dense<0.0045454544> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node947__m.layer-131.kernel" dense<0.00452488707> : tensor<1x1x960x160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node953__m.layer-132.gamma" dense<0.00450450461> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node954__m.layer-132.beta" dense<0.00448430516> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node955__m.layer-132.moving_mean" dense<0.00446428591> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node956__m.layer-132.moving_variance" dense<0.00444444455> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node965__m.layer-134.kernel" dense<0.00442477874> : tensor<1x1x160x960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node971__m.layer-135.gamma" dense<0.00440528616> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node972__m.layer-135.beta" dense<0.00438596494> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node973__m.layer-135.moving_mean" dense<0.0043668123> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node974__m.layer-135.moving_variance" dense<0.00434782589> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node983__m.layer-137.depthwise_kernel" dense<0.00432900432> : tensor<3x3x960x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node989__m.layer-138.gamma" dense<0.00431034481> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node990__m.layer-138.beta" dense<0.00429184549> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node991__m.layer-138.moving_mean" dense<0.00427350448> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node992__m.layer-138.moving_variance" dense<0.00425531901> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1001__m.layer-140.kernel" dense<0.00423728814> : tensor<1x1x960x160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1007__m.layer-141.gamma" dense<0.00421940908> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1008__m.layer-141.beta" dense<0.00420168089> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1009__m.layer-141.moving_mean" dense<0.00418410031> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1010__m.layer-141.moving_variance" dense<0.00416666688> : tensor<160xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1019__m.layer-143.kernel" dense<0.00414937781> : tensor<1x1x160x960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1025__m.layer-144.gamma" dense<0.00413223123> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1026__m.layer-144.beta" dense<0.00411522621> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1027__m.layer-144.moving_mean" dense<0.00409836043> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1028__m.layer-144.moving_variance" dense<0.00408163248> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1037__m.layer-146.depthwise_kernel" dense<0.0040650405> : tensor<3x3x960x1xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1043__m.layer-147.gamma" dense<0.0040485831> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1044__m.layer-147.beta" dense<0.00403225794> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1045__m.layer-147.moving_mean" dense<0.00401606411> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1046__m.layer-147.moving_variance" dense<4.000000e-03> : tensor<960xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1055__m.layer-149.kernel" dense<0.00398406386> : tensor<1x1x960x320xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1061__m.layer-150.gamma" dense<0.0039682542> : tensor<320xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1062__m.layer-150.beta" dense<0.00395256933> : tensor<320xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1063__m.layer-150.moving_mean" dense<0.00393700786> : tensor<320xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1064__m.layer-150.moving_variance" dense<0.00392156886> : tensor<320xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1069__m.layer-151.kernel" dense<3.906250e-03> : tensor<1x1x320x1280xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1075__m.layer-152.gamma" dense<0.00389105058> : tensor<1280xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1076__m.layer-152.beta" dense<0.00387596898> : tensor<1280xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1077__m.layer-152.moving_mean" dense<0.00386100379> : tensor<1280xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1078__m.layer-152.moving_variance" dense<0.00384615385> : tensor<1280xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1091__m.layer-155.kernel" dense<0.00383141753> : tensor<1280x1000xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node1092__m.layer-155.bias" dense<0.00381679391> : tensor<1000xf32> attributes {noinline, sym_visibility = "private"}
  func @predict() attributes { iree.module.export } {
    %arg0 = iree.unfoldable_constant dense<1.5> : tensor<1x224x224x3xf32>
    %0 = flow.variable.address @"__iree_flow___sm_node163__m.layer-1.kernel" : !iree.ptr<tensor<3x3x3x32xf32>>
    %1 = flow.variable.address @"__iree_flow___sm_node169__m.layer-2.gamma" : !iree.ptr<tensor<32xf32>>
    %2 = flow.variable.address @"__iree_flow___sm_node170__m.layer-2.beta" : !iree.ptr<tensor<32xf32>>
    %3 = flow.variable.address @"__iree_flow___sm_node171__m.layer-2.moving_mean" : !iree.ptr<tensor<32xf32>>
    %4 = flow.variable.address @"__iree_flow___sm_node172__m.layer-2.moving_variance" : !iree.ptr<tensor<32xf32>>
    %5 = flow.variable.address @"__iree_flow___sm_node181__m.layer-4.depthwise_kernel" : !iree.ptr<tensor<3x3x32x1xf32>>
    %6 = flow.variable.address @"__iree_flow___sm_node187__m.layer-5.gamma" : !iree.ptr<tensor<32xf32>>
    %7 = flow.variable.address @"__iree_flow___sm_node188__m.layer-5.beta" : !iree.ptr<tensor<32xf32>>
    %8 = flow.variable.address @"__iree_flow___sm_node189__m.layer-5.moving_mean" : !iree.ptr<tensor<32xf32>>
    %9 = flow.variable.address @"__iree_flow___sm_node190__m.layer-5.moving_variance" : !iree.ptr<tensor<32xf32>>
    %10 = flow.variable.address @"__iree_flow___sm_node199__m.layer-7.kernel" : !iree.ptr<tensor<1x1x32x16xf32>>
    %11 = flow.variable.address @"__iree_flow___sm_node205__m.layer-8.gamma" : !iree.ptr<tensor<16xf32>>
    %12 = flow.variable.address @"__iree_flow___sm_node206__m.layer-8.beta" : !iree.ptr<tensor<16xf32>>
    %13 = flow.variable.address @"__iree_flow___sm_node207__m.layer-8.moving_mean" : !iree.ptr<tensor<16xf32>>
    %14 = flow.variable.address @"__iree_flow___sm_node208__m.layer-8.moving_variance" : !iree.ptr<tensor<16xf32>>
    %15 = flow.variable.address @"__iree_flow___sm_node213__m.layer-9.kernel" : !iree.ptr<tensor<1x1x16x96xf32>>
    %16 = flow.variable.address @"__iree_flow___sm_node219__m.layer-10.gamma" : !iree.ptr<tensor<96xf32>>
    %17 = flow.variable.address @"__iree_flow___sm_node220__m.layer-10.beta" : !iree.ptr<tensor<96xf32>>
    %18 = flow.variable.address @"__iree_flow___sm_node221__m.layer-10.moving_mean" : !iree.ptr<tensor<96xf32>>
    %19 = flow.variable.address @"__iree_flow___sm_node222__m.layer-10.moving_variance" : !iree.ptr<tensor<96xf32>>
    %20 = flow.variable.address @"__iree_flow___sm_node235__m.layer-13.depthwise_kernel" : !iree.ptr<tensor<3x3x96x1xf32>>
    %21 = flow.variable.address @"__iree_flow___sm_node241__m.layer-14.gamma" : !iree.ptr<tensor<96xf32>>
    %22 = flow.variable.address @"__iree_flow___sm_node242__m.layer-14.beta" : !iree.ptr<tensor<96xf32>>
    %23 = flow.variable.address @"__iree_flow___sm_node243__m.layer-14.moving_mean" : !iree.ptr<tensor<96xf32>>
    %24 = flow.variable.address @"__iree_flow___sm_node244__m.layer-14.moving_variance" : !iree.ptr<tensor<96xf32>>
    %25 = flow.variable.address @"__iree_flow___sm_node253__m.layer-16.kernel" : !iree.ptr<tensor<1x1x96x24xf32>>
    %26 = flow.variable.address @"__iree_flow___sm_node259__m.layer-17.gamma" : !iree.ptr<tensor<24xf32>>
    %27 = flow.variable.address @"__iree_flow___sm_node260__m.layer-17.beta" : !iree.ptr<tensor<24xf32>>
    %28 = flow.variable.address @"__iree_flow___sm_node261__m.layer-17.moving_mean" : !iree.ptr<tensor<24xf32>>
    %29 = flow.variable.address @"__iree_flow___sm_node262__m.layer-17.moving_variance" : !iree.ptr<tensor<24xf32>>
    %30 = flow.variable.address @"__iree_flow___sm_node267__m.layer-18.kernel" : !iree.ptr<tensor<1x1x24x144xf32>>
    %31 = flow.variable.address @"__iree_flow___sm_node273__m.layer-19.gamma" : !iree.ptr<tensor<144xf32>>
    %32 = flow.variable.address @"__iree_flow___sm_node274__m.layer-19.beta" : !iree.ptr<tensor<144xf32>>
    %33 = flow.variable.address @"__iree_flow___sm_node275__m.layer-19.moving_mean" : !iree.ptr<tensor<144xf32>>
    %34 = flow.variable.address @"__iree_flow___sm_node276__m.layer-19.moving_variance" : !iree.ptr<tensor<144xf32>>
    %35 = flow.variable.address @"__iree_flow___sm_node285__m.layer-21.depthwise_kernel" : !iree.ptr<tensor<3x3x144x1xf32>>
    %36 = flow.variable.address @"__iree_flow___sm_node291__m.layer-22.gamma" : !iree.ptr<tensor<144xf32>>
    %37 = flow.variable.address @"__iree_flow___sm_node292__m.layer-22.beta" : !iree.ptr<tensor<144xf32>>
    %38 = flow.variable.address @"__iree_flow___sm_node293__m.layer-22.moving_mean" : !iree.ptr<tensor<144xf32>>
    %39 = flow.variable.address @"__iree_flow___sm_node294__m.layer-22.moving_variance" : !iree.ptr<tensor<144xf32>>
    %40 = flow.variable.address @"__iree_flow___sm_node303__m.layer-24.kernel" : !iree.ptr<tensor<1x1x144x24xf32>>
    %41 = flow.variable.address @"__iree_flow___sm_node309__m.layer-25.gamma" : !iree.ptr<tensor<24xf32>>
    %42 = flow.variable.address @"__iree_flow___sm_node310__m.layer-25.beta" : !iree.ptr<tensor<24xf32>>
    %43 = flow.variable.address @"__iree_flow___sm_node311__m.layer-25.moving_mean" : !iree.ptr<tensor<24xf32>>
    %44 = flow.variable.address @"__iree_flow___sm_node312__m.layer-25.moving_variance" : !iree.ptr<tensor<24xf32>>
    %45 = flow.variable.address @"__iree_flow___sm_node321__m.layer-27.kernel" : !iree.ptr<tensor<1x1x24x144xf32>>
    %46 = flow.variable.address @"__iree_flow___sm_node327__m.layer-28.gamma" : !iree.ptr<tensor<144xf32>>
    %47 = flow.variable.address @"__iree_flow___sm_node328__m.layer-28.beta" : !iree.ptr<tensor<144xf32>>
    %48 = flow.variable.address @"__iree_flow___sm_node329__m.layer-28.moving_mean" : !iree.ptr<tensor<144xf32>>
    %49 = flow.variable.address @"__iree_flow___sm_node330__m.layer-28.moving_variance" : !iree.ptr<tensor<144xf32>>
    %50 = flow.variable.address @"__iree_flow___sm_node343__m.layer-31.depthwise_kernel" : !iree.ptr<tensor<3x3x144x1xf32>>
    %51 = flow.variable.address @"__iree_flow___sm_node349__m.layer-32.gamma" : !iree.ptr<tensor<144xf32>>
    %52 = flow.variable.address @"__iree_flow___sm_node350__m.layer-32.beta" : !iree.ptr<tensor<144xf32>>
    %53 = flow.variable.address @"__iree_flow___sm_node351__m.layer-32.moving_mean" : !iree.ptr<tensor<144xf32>>
    %54 = flow.variable.address @"__iree_flow___sm_node352__m.layer-32.moving_variance" : !iree.ptr<tensor<144xf32>>
    %55 = flow.variable.address @"__iree_flow___sm_node361__m.layer-34.kernel" : !iree.ptr<tensor<1x1x144x32xf32>>
    %56 = flow.variable.address @"__iree_flow___sm_node367__m.layer-35.gamma" : !iree.ptr<tensor<32xf32>>
    %57 = flow.variable.address @"__iree_flow___sm_node368__m.layer-35.beta" : !iree.ptr<tensor<32xf32>>
    %58 = flow.variable.address @"__iree_flow___sm_node369__m.layer-35.moving_mean" : !iree.ptr<tensor<32xf32>>
    %59 = flow.variable.address @"__iree_flow___sm_node370__m.layer-35.moving_variance" : !iree.ptr<tensor<32xf32>>
    %60 = flow.variable.address @"__iree_flow___sm_node375__m.layer-36.kernel" : !iree.ptr<tensor<1x1x32x192xf32>>
    %61 = flow.variable.address @"__iree_flow___sm_node381__m.layer-37.gamma" : !iree.ptr<tensor<192xf32>>
    %62 = flow.variable.address @"__iree_flow___sm_node382__m.layer-37.beta" : !iree.ptr<tensor<192xf32>>
    %63 = flow.variable.address @"__iree_flow___sm_node383__m.layer-37.moving_mean" : !iree.ptr<tensor<192xf32>>
    %64 = flow.variable.address @"__iree_flow___sm_node384__m.layer-37.moving_variance" : !iree.ptr<tensor<192xf32>>
    %65 = flow.variable.address @"__iree_flow___sm_node393__m.layer-39.depthwise_kernel" : !iree.ptr<tensor<3x3x192x1xf32>>
    %66 = flow.variable.address @"__iree_flow___sm_node399__m.layer-40.gamma" : !iree.ptr<tensor<192xf32>>
    %67 = flow.variable.address @"__iree_flow___sm_node400__m.layer-40.beta" : !iree.ptr<tensor<192xf32>>
    %68 = flow.variable.address @"__iree_flow___sm_node401__m.layer-40.moving_mean" : !iree.ptr<tensor<192xf32>>
    %69 = flow.variable.address @"__iree_flow___sm_node402__m.layer-40.moving_variance" : !iree.ptr<tensor<192xf32>>
    %70 = flow.variable.address @"__iree_flow___sm_node411__m.layer-42.kernel" : !iree.ptr<tensor<1x1x192x32xf32>>
    %71 = flow.variable.address @"__iree_flow___sm_node417__m.layer-43.gamma" : !iree.ptr<tensor<32xf32>>
    %72 = flow.variable.address @"__iree_flow___sm_node418__m.layer-43.beta" : !iree.ptr<tensor<32xf32>>
    %73 = flow.variable.address @"__iree_flow___sm_node419__m.layer-43.moving_mean" : !iree.ptr<tensor<32xf32>>
    %74 = flow.variable.address @"__iree_flow___sm_node420__m.layer-43.moving_variance" : !iree.ptr<tensor<32xf32>>
    %75 = flow.variable.address @"__iree_flow___sm_node429__m.layer-45.kernel" : !iree.ptr<tensor<1x1x32x192xf32>>
    %76 = flow.variable.address @"__iree_flow___sm_node435__m.layer-46.gamma" : !iree.ptr<tensor<192xf32>>
    %77 = flow.variable.address @"__iree_flow___sm_node436__m.layer-46.beta" : !iree.ptr<tensor<192xf32>>
    %78 = flow.variable.address @"__iree_flow___sm_node437__m.layer-46.moving_mean" : !iree.ptr<tensor<192xf32>>
    %79 = flow.variable.address @"__iree_flow___sm_node438__m.layer-46.moving_variance" : !iree.ptr<tensor<192xf32>>
    %80 = flow.variable.address @"__iree_flow___sm_node447__m.layer-48.depthwise_kernel" : !iree.ptr<tensor<3x3x192x1xf32>>
    %81 = flow.variable.address @"__iree_flow___sm_node453__m.layer-49.gamma" : !iree.ptr<tensor<192xf32>>
    %82 = flow.variable.address @"__iree_flow___sm_node454__m.layer-49.beta" : !iree.ptr<tensor<192xf32>>
    %83 = flow.variable.address @"__iree_flow___sm_node455__m.layer-49.moving_mean" : !iree.ptr<tensor<192xf32>>
    %84 = flow.variable.address @"__iree_flow___sm_node456__m.layer-49.moving_variance" : !iree.ptr<tensor<192xf32>>
    %85 = flow.variable.address @"__iree_flow___sm_node465__m.layer-51.kernel" : !iree.ptr<tensor<1x1x192x32xf32>>
    %86 = flow.variable.address @"__iree_flow___sm_node471__m.layer-52.gamma" : !iree.ptr<tensor<32xf32>>
    %87 = flow.variable.address @"__iree_flow___sm_node472__m.layer-52.beta" : !iree.ptr<tensor<32xf32>>
    %88 = flow.variable.address @"__iree_flow___sm_node473__m.layer-52.moving_mean" : !iree.ptr<tensor<32xf32>>
    %89 = flow.variable.address @"__iree_flow___sm_node474__m.layer-52.moving_variance" : !iree.ptr<tensor<32xf32>>
    %90 = flow.variable.address @"__iree_flow___sm_node483__m.layer-54.kernel" : !iree.ptr<tensor<1x1x32x192xf32>>
    %91 = flow.variable.address @"__iree_flow___sm_node489__m.layer-55.gamma" : !iree.ptr<tensor<192xf32>>
    %92 = flow.variable.address @"__iree_flow___sm_node490__m.layer-55.beta" : !iree.ptr<tensor<192xf32>>
    %93 = flow.variable.address @"__iree_flow___sm_node491__m.layer-55.moving_mean" : !iree.ptr<tensor<192xf32>>
    %94 = flow.variable.address @"__iree_flow___sm_node492__m.layer-55.moving_variance" : !iree.ptr<tensor<192xf32>>
    %95 = flow.variable.address @"__iree_flow___sm_node505__m.layer-58.depthwise_kernel" : !iree.ptr<tensor<3x3x192x1xf32>>
    %96 = flow.variable.address @"__iree_flow___sm_node511__m.layer-59.gamma" : !iree.ptr<tensor<192xf32>>
    %97 = flow.variable.address @"__iree_flow___sm_node512__m.layer-59.beta" : !iree.ptr<tensor<192xf32>>
    %98 = flow.variable.address @"__iree_flow___sm_node513__m.layer-59.moving_mean" : !iree.ptr<tensor<192xf32>>
    %99 = flow.variable.address @"__iree_flow___sm_node514__m.layer-59.moving_variance" : !iree.ptr<tensor<192xf32>>
    %100 = flow.variable.address @"__iree_flow___sm_node523__m.layer-61.kernel" : !iree.ptr<tensor<1x1x192x64xf32>>
    %101 = flow.variable.address @"__iree_flow___sm_node529__m.layer-62.gamma" : !iree.ptr<tensor<64xf32>>
    %102 = flow.variable.address @"__iree_flow___sm_node530__m.layer-62.beta" : !iree.ptr<tensor<64xf32>>
    %103 = flow.variable.address @"__iree_flow___sm_node531__m.layer-62.moving_mean" : !iree.ptr<tensor<64xf32>>
    %104 = flow.variable.address @"__iree_flow___sm_node532__m.layer-62.moving_variance" : !iree.ptr<tensor<64xf32>>
    %105 = flow.variable.address @"__iree_flow___sm_node537__m.layer-63.kernel" : !iree.ptr<tensor<1x1x64x384xf32>>
    %106 = flow.variable.address @"__iree_flow___sm_node543__m.layer-64.gamma" : !iree.ptr<tensor<384xf32>>
    %107 = flow.variable.address @"__iree_flow___sm_node544__m.layer-64.beta" : !iree.ptr<tensor<384xf32>>
    %108 = flow.variable.address @"__iree_flow___sm_node545__m.layer-64.moving_mean" : !iree.ptr<tensor<384xf32>>
    %109 = flow.variable.address @"__iree_flow___sm_node546__m.layer-64.moving_variance" : !iree.ptr<tensor<384xf32>>
    %110 = flow.variable.address @"__iree_flow___sm_node555__m.layer-66.depthwise_kernel" : !iree.ptr<tensor<3x3x384x1xf32>>
    %111 = flow.variable.address @"__iree_flow___sm_node561__m.layer-67.gamma" : !iree.ptr<tensor<384xf32>>
    %112 = flow.variable.address @"__iree_flow___sm_node562__m.layer-67.beta" : !iree.ptr<tensor<384xf32>>
    %113 = flow.variable.address @"__iree_flow___sm_node563__m.layer-67.moving_mean" : !iree.ptr<tensor<384xf32>>
    %114 = flow.variable.address @"__iree_flow___sm_node564__m.layer-67.moving_variance" : !iree.ptr<tensor<384xf32>>
    %115 = flow.variable.address @"__iree_flow___sm_node573__m.layer-69.kernel" : !iree.ptr<tensor<1x1x384x64xf32>>
    %116 = flow.variable.address @"__iree_flow___sm_node579__m.layer-70.gamma" : !iree.ptr<tensor<64xf32>>
    %117 = flow.variable.address @"__iree_flow___sm_node580__m.layer-70.beta" : !iree.ptr<tensor<64xf32>>
    %118 = flow.variable.address @"__iree_flow___sm_node581__m.layer-70.moving_mean" : !iree.ptr<tensor<64xf32>>
    %119 = flow.variable.address @"__iree_flow___sm_node582__m.layer-70.moving_variance" : !iree.ptr<tensor<64xf32>>
    %120 = flow.variable.address @"__iree_flow___sm_node591__m.layer-72.kernel" : !iree.ptr<tensor<1x1x64x384xf32>>
    %121 = flow.variable.address @"__iree_flow___sm_node597__m.layer-73.gamma" : !iree.ptr<tensor<384xf32>>
    %122 = flow.variable.address @"__iree_flow___sm_node598__m.layer-73.beta" : !iree.ptr<tensor<384xf32>>
    %123 = flow.variable.address @"__iree_flow___sm_node599__m.layer-73.moving_mean" : !iree.ptr<tensor<384xf32>>
    %124 = flow.variable.address @"__iree_flow___sm_node600__m.layer-73.moving_variance" : !iree.ptr<tensor<384xf32>>
    %125 = flow.variable.address @"__iree_flow___sm_node609__m.layer-75.depthwise_kernel" : !iree.ptr<tensor<3x3x384x1xf32>>
    %126 = flow.variable.address @"__iree_flow___sm_node615__m.layer-76.gamma" : !iree.ptr<tensor<384xf32>>
    %127 = flow.variable.address @"__iree_flow___sm_node616__m.layer-76.beta" : !iree.ptr<tensor<384xf32>>
    %128 = flow.variable.address @"__iree_flow___sm_node617__m.layer-76.moving_mean" : !iree.ptr<tensor<384xf32>>
    %129 = flow.variable.address @"__iree_flow___sm_node618__m.layer-76.moving_variance" : !iree.ptr<tensor<384xf32>>
    %130 = flow.variable.address @"__iree_flow___sm_node627__m.layer-78.kernel" : !iree.ptr<tensor<1x1x384x64xf32>>
    %131 = flow.variable.address @"__iree_flow___sm_node633__m.layer-79.gamma" : !iree.ptr<tensor<64xf32>>
    %132 = flow.variable.address @"__iree_flow___sm_node634__m.layer-79.beta" : !iree.ptr<tensor<64xf32>>
    %133 = flow.variable.address @"__iree_flow___sm_node635__m.layer-79.moving_mean" : !iree.ptr<tensor<64xf32>>
    %134 = flow.variable.address @"__iree_flow___sm_node636__m.layer-79.moving_variance" : !iree.ptr<tensor<64xf32>>
    %135 = flow.variable.address @"__iree_flow___sm_node645__m.layer-81.kernel" : !iree.ptr<tensor<1x1x64x384xf32>>
    %136 = flow.variable.address @"__iree_flow___sm_node651__m.layer-82.gamma" : !iree.ptr<tensor<384xf32>>
    %137 = flow.variable.address @"__iree_flow___sm_node652__m.layer-82.beta" : !iree.ptr<tensor<384xf32>>
    %138 = flow.variable.address @"__iree_flow___sm_node653__m.layer-82.moving_mean" : !iree.ptr<tensor<384xf32>>
    %139 = flow.variable.address @"__iree_flow___sm_node654__m.layer-82.moving_variance" : !iree.ptr<tensor<384xf32>>
    %140 = flow.variable.address @"__iree_flow___sm_node663__m.layer-84.depthwise_kernel" : !iree.ptr<tensor<3x3x384x1xf32>>
    %141 = flow.variable.address @"__iree_flow___sm_node669__m.layer-85.gamma" : !iree.ptr<tensor<384xf32>>
    %142 = flow.variable.address @"__iree_flow___sm_node670__m.layer-85.beta" : !iree.ptr<tensor<384xf32>>
    %143 = flow.variable.address @"__iree_flow___sm_node671__m.layer-85.moving_mean" : !iree.ptr<tensor<384xf32>>
    %144 = flow.variable.address @"__iree_flow___sm_node672__m.layer-85.moving_variance" : !iree.ptr<tensor<384xf32>>
    %145 = flow.variable.address @"__iree_flow___sm_node681__m.layer-87.kernel" : !iree.ptr<tensor<1x1x384x64xf32>>
    %146 = flow.variable.address @"__iree_flow___sm_node687__m.layer-88.gamma" : !iree.ptr<tensor<64xf32>>
    %147 = flow.variable.address @"__iree_flow___sm_node688__m.layer-88.beta" : !iree.ptr<tensor<64xf32>>
    %148 = flow.variable.address @"__iree_flow___sm_node689__m.layer-88.moving_mean" : !iree.ptr<tensor<64xf32>>
    %149 = flow.variable.address @"__iree_flow___sm_node690__m.layer-88.moving_variance" : !iree.ptr<tensor<64xf32>>
    %150 = flow.variable.address @"__iree_flow___sm_node699__m.layer-90.kernel" : !iree.ptr<tensor<1x1x64x384xf32>>
    %151 = flow.variable.address @"__iree_flow___sm_node705__m.layer-91.gamma" : !iree.ptr<tensor<384xf32>>
    %152 = flow.variable.address @"__iree_flow___sm_node706__m.layer-91.beta" : !iree.ptr<tensor<384xf32>>
    %153 = flow.variable.address @"__iree_flow___sm_node707__m.layer-91.moving_mean" : !iree.ptr<tensor<384xf32>>
    %154 = flow.variable.address @"__iree_flow___sm_node708__m.layer-91.moving_variance" : !iree.ptr<tensor<384xf32>>
    %155 = flow.variable.address @"__iree_flow___sm_node717__m.layer-93.depthwise_kernel" : !iree.ptr<tensor<3x3x384x1xf32>>
    %156 = flow.variable.address @"__iree_flow___sm_node723__m.layer-94.gamma" : !iree.ptr<tensor<384xf32>>
    %157 = flow.variable.address @"__iree_flow___sm_node724__m.layer-94.beta" : !iree.ptr<tensor<384xf32>>
    %158 = flow.variable.address @"__iree_flow___sm_node725__m.layer-94.moving_mean" : !iree.ptr<tensor<384xf32>>
    %159 = flow.variable.address @"__iree_flow___sm_node726__m.layer-94.moving_variance" : !iree.ptr<tensor<384xf32>>
    %160 = flow.variable.address @"__iree_flow___sm_node735__m.layer-96.kernel" : !iree.ptr<tensor<1x1x384x96xf32>>
    %161 = flow.variable.address @"__iree_flow___sm_node741__m.layer-97.gamma" : !iree.ptr<tensor<96xf32>>
    %162 = flow.variable.address @"__iree_flow___sm_node742__m.layer-97.beta" : !iree.ptr<tensor<96xf32>>
    %163 = flow.variable.address @"__iree_flow___sm_node743__m.layer-97.moving_mean" : !iree.ptr<tensor<96xf32>>
    %164 = flow.variable.address @"__iree_flow___sm_node744__m.layer-97.moving_variance" : !iree.ptr<tensor<96xf32>>
    %165 = flow.variable.address @"__iree_flow___sm_node749__m.layer-98.kernel" : !iree.ptr<tensor<1x1x96x576xf32>>
    %166 = flow.variable.address @"__iree_flow___sm_node755__m.layer-99.gamma" : !iree.ptr<tensor<576xf32>>
    %167 = flow.variable.address @"__iree_flow___sm_node756__m.layer-99.beta" : !iree.ptr<tensor<576xf32>>
    %168 = flow.variable.address @"__iree_flow___sm_node757__m.layer-99.moving_mean" : !iree.ptr<tensor<576xf32>>
    %169 = flow.variable.address @"__iree_flow___sm_node758__m.layer-99.moving_variance" : !iree.ptr<tensor<576xf32>>
    %170 = flow.variable.address @"__iree_flow___sm_node767__m.layer-101.depthwise_kernel" : !iree.ptr<tensor<3x3x576x1xf32>>
    %171 = flow.variable.address @"__iree_flow___sm_node773__m.layer-102.gamma" : !iree.ptr<tensor<576xf32>>
    %172 = flow.variable.address @"__iree_flow___sm_node774__m.layer-102.beta" : !iree.ptr<tensor<576xf32>>
    %173 = flow.variable.address @"__iree_flow___sm_node775__m.layer-102.moving_mean" : !iree.ptr<tensor<576xf32>>
    %174 = flow.variable.address @"__iree_flow___sm_node776__m.layer-102.moving_variance" : !iree.ptr<tensor<576xf32>>
    %175 = flow.variable.address @"__iree_flow___sm_node785__m.layer-104.kernel" : !iree.ptr<tensor<1x1x576x96xf32>>
    %176 = flow.variable.address @"__iree_flow___sm_node791__m.layer-105.gamma" : !iree.ptr<tensor<96xf32>>
    %177 = flow.variable.address @"__iree_flow___sm_node792__m.layer-105.beta" : !iree.ptr<tensor<96xf32>>
    %178 = flow.variable.address @"__iree_flow___sm_node793__m.layer-105.moving_mean" : !iree.ptr<tensor<96xf32>>
    %179 = flow.variable.address @"__iree_flow___sm_node794__m.layer-105.moving_variance" : !iree.ptr<tensor<96xf32>>
    %180 = flow.variable.address @"__iree_flow___sm_node803__m.layer-107.kernel" : !iree.ptr<tensor<1x1x96x576xf32>>
    %181 = flow.variable.address @"__iree_flow___sm_node809__m.layer-108.gamma" : !iree.ptr<tensor<576xf32>>
    %182 = flow.variable.address @"__iree_flow___sm_node810__m.layer-108.beta" : !iree.ptr<tensor<576xf32>>
    %183 = flow.variable.address @"__iree_flow___sm_node811__m.layer-108.moving_mean" : !iree.ptr<tensor<576xf32>>
    %184 = flow.variable.address @"__iree_flow___sm_node812__m.layer-108.moving_variance" : !iree.ptr<tensor<576xf32>>
    %185 = flow.variable.address @"__iree_flow___sm_node821__m.layer-110.depthwise_kernel" : !iree.ptr<tensor<3x3x576x1xf32>>
    %186 = flow.variable.address @"__iree_flow___sm_node827__m.layer-111.gamma" : !iree.ptr<tensor<576xf32>>
    %187 = flow.variable.address @"__iree_flow___sm_node828__m.layer-111.beta" : !iree.ptr<tensor<576xf32>>
    %188 = flow.variable.address @"__iree_flow___sm_node829__m.layer-111.moving_mean" : !iree.ptr<tensor<576xf32>>
    %189 = flow.variable.address @"__iree_flow___sm_node830__m.layer-111.moving_variance" : !iree.ptr<tensor<576xf32>>
    %190 = flow.variable.address @"__iree_flow___sm_node839__m.layer-113.kernel" : !iree.ptr<tensor<1x1x576x96xf32>>
    %191 = flow.variable.address @"__iree_flow___sm_node845__m.layer-114.gamma" : !iree.ptr<tensor<96xf32>>
    %192 = flow.variable.address @"__iree_flow___sm_node846__m.layer-114.beta" : !iree.ptr<tensor<96xf32>>
    %193 = flow.variable.address @"__iree_flow___sm_node847__m.layer-114.moving_mean" : !iree.ptr<tensor<96xf32>>
    %194 = flow.variable.address @"__iree_flow___sm_node848__m.layer-114.moving_variance" : !iree.ptr<tensor<96xf32>>
    %195 = flow.variable.address @"__iree_flow___sm_node857__m.layer-116.kernel" : !iree.ptr<tensor<1x1x96x576xf32>>
    %196 = flow.variable.address @"__iree_flow___sm_node863__m.layer-117.gamma" : !iree.ptr<tensor<576xf32>>
    %197 = flow.variable.address @"__iree_flow___sm_node864__m.layer-117.beta" : !iree.ptr<tensor<576xf32>>
    %198 = flow.variable.address @"__iree_flow___sm_node865__m.layer-117.moving_mean" : !iree.ptr<tensor<576xf32>>
    %199 = flow.variable.address @"__iree_flow___sm_node866__m.layer-117.moving_variance" : !iree.ptr<tensor<576xf32>>
    %200 = flow.variable.address @"__iree_flow___sm_node879__m.layer-120.depthwise_kernel" : !iree.ptr<tensor<3x3x576x1xf32>>
    %201 = flow.variable.address @"__iree_flow___sm_node885__m.layer-121.gamma" : !iree.ptr<tensor<576xf32>>
    %202 = flow.variable.address @"__iree_flow___sm_node886__m.layer-121.beta" : !iree.ptr<tensor<576xf32>>
    %203 = flow.variable.address @"__iree_flow___sm_node887__m.layer-121.moving_mean" : !iree.ptr<tensor<576xf32>>
    %204 = flow.variable.address @"__iree_flow___sm_node888__m.layer-121.moving_variance" : !iree.ptr<tensor<576xf32>>
    %205 = flow.variable.address @"__iree_flow___sm_node897__m.layer-123.kernel" : !iree.ptr<tensor<1x1x576x160xf32>>
    %206 = flow.variable.address @"__iree_flow___sm_node903__m.layer-124.gamma" : !iree.ptr<tensor<160xf32>>
    %207 = flow.variable.address @"__iree_flow___sm_node904__m.layer-124.beta" : !iree.ptr<tensor<160xf32>>
    %208 = flow.variable.address @"__iree_flow___sm_node905__m.layer-124.moving_mean" : !iree.ptr<tensor<160xf32>>
    %209 = flow.variable.address @"__iree_flow___sm_node906__m.layer-124.moving_variance" : !iree.ptr<tensor<160xf32>>
    %210 = flow.variable.address @"__iree_flow___sm_node911__m.layer-125.kernel" : !iree.ptr<tensor<1x1x160x960xf32>>
    %211 = flow.variable.address @"__iree_flow___sm_node917__m.layer-126.gamma" : !iree.ptr<tensor<960xf32>>
    %212 = flow.variable.address @"__iree_flow___sm_node918__m.layer-126.beta" : !iree.ptr<tensor<960xf32>>
    %213 = flow.variable.address @"__iree_flow___sm_node919__m.layer-126.moving_mean" : !iree.ptr<tensor<960xf32>>
    %214 = flow.variable.address @"__iree_flow___sm_node920__m.layer-126.moving_variance" : !iree.ptr<tensor<960xf32>>
    %215 = flow.variable.address @"__iree_flow___sm_node929__m.layer-128.depthwise_kernel" : !iree.ptr<tensor<3x3x960x1xf32>>
    %216 = flow.variable.address @"__iree_flow___sm_node935__m.layer-129.gamma" : !iree.ptr<tensor<960xf32>>
    %217 = flow.variable.address @"__iree_flow___sm_node936__m.layer-129.beta" : !iree.ptr<tensor<960xf32>>
    %218 = flow.variable.address @"__iree_flow___sm_node937__m.layer-129.moving_mean" : !iree.ptr<tensor<960xf32>>
    %219 = flow.variable.address @"__iree_flow___sm_node938__m.layer-129.moving_variance" : !iree.ptr<tensor<960xf32>>
    %220 = flow.variable.address @"__iree_flow___sm_node947__m.layer-131.kernel" : !iree.ptr<tensor<1x1x960x160xf32>>
    %221 = flow.variable.address @"__iree_flow___sm_node953__m.layer-132.gamma" : !iree.ptr<tensor<160xf32>>
    %222 = flow.variable.address @"__iree_flow___sm_node954__m.layer-132.beta" : !iree.ptr<tensor<160xf32>>
    %223 = flow.variable.address @"__iree_flow___sm_node955__m.layer-132.moving_mean" : !iree.ptr<tensor<160xf32>>
    %224 = flow.variable.address @"__iree_flow___sm_node956__m.layer-132.moving_variance" : !iree.ptr<tensor<160xf32>>
    %225 = flow.variable.address @"__iree_flow___sm_node965__m.layer-134.kernel" : !iree.ptr<tensor<1x1x160x960xf32>>
    %226 = flow.variable.address @"__iree_flow___sm_node971__m.layer-135.gamma" : !iree.ptr<tensor<960xf32>>
    %227 = flow.variable.address @"__iree_flow___sm_node972__m.layer-135.beta" : !iree.ptr<tensor<960xf32>>
    %228 = flow.variable.address @"__iree_flow___sm_node973__m.layer-135.moving_mean" : !iree.ptr<tensor<960xf32>>
    %229 = flow.variable.address @"__iree_flow___sm_node974__m.layer-135.moving_variance" : !iree.ptr<tensor<960xf32>>
    %230 = flow.variable.address @"__iree_flow___sm_node983__m.layer-137.depthwise_kernel" : !iree.ptr<tensor<3x3x960x1xf32>>
    %231 = flow.variable.address @"__iree_flow___sm_node989__m.layer-138.gamma" : !iree.ptr<tensor<960xf32>>
    %232 = flow.variable.address @"__iree_flow___sm_node990__m.layer-138.beta" : !iree.ptr<tensor<960xf32>>
    %233 = flow.variable.address @"__iree_flow___sm_node991__m.layer-138.moving_mean" : !iree.ptr<tensor<960xf32>>
    %234 = flow.variable.address @"__iree_flow___sm_node992__m.layer-138.moving_variance" : !iree.ptr<tensor<960xf32>>
    %235 = flow.variable.address @"__iree_flow___sm_node1001__m.layer-140.kernel" : !iree.ptr<tensor<1x1x960x160xf32>>
    %236 = flow.variable.address @"__iree_flow___sm_node1007__m.layer-141.gamma" : !iree.ptr<tensor<160xf32>>
    %237 = flow.variable.address @"__iree_flow___sm_node1008__m.layer-141.beta" : !iree.ptr<tensor<160xf32>>
    %238 = flow.variable.address @"__iree_flow___sm_node1009__m.layer-141.moving_mean" : !iree.ptr<tensor<160xf32>>
    %239 = flow.variable.address @"__iree_flow___sm_node1010__m.layer-141.moving_variance" : !iree.ptr<tensor<160xf32>>
    %240 = flow.variable.address @"__iree_flow___sm_node1019__m.layer-143.kernel" : !iree.ptr<tensor<1x1x160x960xf32>>
    %241 = flow.variable.address @"__iree_flow___sm_node1025__m.layer-144.gamma" : !iree.ptr<tensor<960xf32>>
    %242 = flow.variable.address @"__iree_flow___sm_node1026__m.layer-144.beta" : !iree.ptr<tensor<960xf32>>
    %243 = flow.variable.address @"__iree_flow___sm_node1027__m.layer-144.moving_mean" : !iree.ptr<tensor<960xf32>>
    %244 = flow.variable.address @"__iree_flow___sm_node1028__m.layer-144.moving_variance" : !iree.ptr<tensor<960xf32>>
    %245 = flow.variable.address @"__iree_flow___sm_node1037__m.layer-146.depthwise_kernel" : !iree.ptr<tensor<3x3x960x1xf32>>
    %246 = flow.variable.address @"__iree_flow___sm_node1043__m.layer-147.gamma" : !iree.ptr<tensor<960xf32>>
    %247 = flow.variable.address @"__iree_flow___sm_node1044__m.layer-147.beta" : !iree.ptr<tensor<960xf32>>
    %248 = flow.variable.address @"__iree_flow___sm_node1045__m.layer-147.moving_mean" : !iree.ptr<tensor<960xf32>>
    %249 = flow.variable.address @"__iree_flow___sm_node1046__m.layer-147.moving_variance" : !iree.ptr<tensor<960xf32>>
    %250 = flow.variable.address @"__iree_flow___sm_node1055__m.layer-149.kernel" : !iree.ptr<tensor<1x1x960x320xf32>>
    %251 = flow.variable.address @"__iree_flow___sm_node1061__m.layer-150.gamma" : !iree.ptr<tensor<320xf32>>
    %252 = flow.variable.address @"__iree_flow___sm_node1062__m.layer-150.beta" : !iree.ptr<tensor<320xf32>>
    %253 = flow.variable.address @"__iree_flow___sm_node1063__m.layer-150.moving_mean" : !iree.ptr<tensor<320xf32>>
    %254 = flow.variable.address @"__iree_flow___sm_node1064__m.layer-150.moving_variance" : !iree.ptr<tensor<320xf32>>
    %255 = flow.variable.address @"__iree_flow___sm_node1069__m.layer-151.kernel" : !iree.ptr<tensor<1x1x320x1280xf32>>
    %256 = flow.variable.address @"__iree_flow___sm_node1075__m.layer-152.gamma" : !iree.ptr<tensor<1280xf32>>
    %257 = flow.variable.address @"__iree_flow___sm_node1076__m.layer-152.beta" : !iree.ptr<tensor<1280xf32>>
    %258 = flow.variable.address @"__iree_flow___sm_node1077__m.layer-152.moving_mean" : !iree.ptr<tensor<1280xf32>>
    %259 = flow.variable.address @"__iree_flow___sm_node1078__m.layer-152.moving_variance" : !iree.ptr<tensor<1280xf32>>
    %260 = flow.variable.address @"__iree_flow___sm_node1091__m.layer-155.kernel" : !iree.ptr<tensor<1280x1000xf32>>
    %261 = flow.variable.address @"__iree_flow___sm_node1092__m.layer-155.bias" : !iree.ptr<tensor<1000xf32>>
    %262 = mhlo.constant dense<6.000000e+00> : tensor<f32>
    %263 = mhlo.constant dense<4.900000e+01> : tensor<1x1280xf32>
    %264 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %265 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %266 = flow.variable.load.indirect %159 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %267 = flow.variable.load.indirect %158 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %268 = flow.variable.load.indirect %157 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %269 = flow.variable.load.indirect %156 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %270 = flow.variable.load.indirect %155 : !iree.ptr<tensor<3x3x384x1xf32>> -> tensor<3x3x384x1xf32>
    %271 = flow.variable.load.indirect %154 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %272 = flow.variable.load.indirect %153 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %273 = flow.variable.load.indirect %152 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %274 = flow.variable.load.indirect %151 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %275 = flow.variable.load.indirect %150 : !iree.ptr<tensor<1x1x64x384xf32>> -> tensor<1x1x64x384xf32>
    %276 = flow.variable.load.indirect %164 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %277 = flow.variable.load.indirect %163 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %278 = flow.variable.load.indirect %162 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %279 = flow.variable.load.indirect %161 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %280 = flow.variable.load.indirect %160 : !iree.ptr<tensor<1x1x384x96xf32>> -> tensor<1x1x384x96xf32>
    %281 = flow.variable.load.indirect %174 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %282 = flow.variable.load.indirect %173 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %283 = flow.variable.load.indirect %172 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %284 = flow.variable.load.indirect %171 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %285 = flow.variable.load.indirect %170 : !iree.ptr<tensor<3x3x576x1xf32>> -> tensor<3x3x576x1xf32>
    %286 = flow.variable.load.indirect %169 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %287 = flow.variable.load.indirect %168 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %288 = flow.variable.load.indirect %167 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %289 = flow.variable.load.indirect %166 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %290 = flow.variable.load.indirect %165 : !iree.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %291 = flow.variable.load.indirect %179 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %292 = flow.variable.load.indirect %178 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %293 = flow.variable.load.indirect %177 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %294 = flow.variable.load.indirect %176 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %295 = flow.variable.load.indirect %175 : !iree.ptr<tensor<1x1x576x96xf32>> -> tensor<1x1x576x96xf32>
    %296 = flow.variable.load.indirect %189 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %297 = flow.variable.load.indirect %188 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %298 = flow.variable.load.indirect %187 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %299 = flow.variable.load.indirect %186 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %300 = flow.variable.load.indirect %185 : !iree.ptr<tensor<3x3x576x1xf32>> -> tensor<3x3x576x1xf32>
    %301 = flow.variable.load.indirect %184 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %302 = flow.variable.load.indirect %183 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %303 = flow.variable.load.indirect %182 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %304 = flow.variable.load.indirect %181 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %305 = flow.variable.load.indirect %180 : !iree.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %306 = flow.variable.load.indirect %194 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %307 = flow.variable.load.indirect %193 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %308 = flow.variable.load.indirect %192 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %309 = flow.variable.load.indirect %191 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %310 = flow.variable.load.indirect %190 : !iree.ptr<tensor<1x1x576x96xf32>> -> tensor<1x1x576x96xf32>
    %311 = flow.variable.load.indirect %204 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %312 = flow.variable.load.indirect %203 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %313 = flow.variable.load.indirect %202 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %314 = flow.variable.load.indirect %201 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %315 = flow.variable.load.indirect %200 : !iree.ptr<tensor<3x3x576x1xf32>> -> tensor<3x3x576x1xf32>
    %316 = flow.variable.load.indirect %199 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %317 = flow.variable.load.indirect %198 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %318 = flow.variable.load.indirect %197 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %319 = flow.variable.load.indirect %196 : !iree.ptr<tensor<576xf32>> -> tensor<576xf32>
    %320 = flow.variable.load.indirect %195 : !iree.ptr<tensor<1x1x96x576xf32>> -> tensor<1x1x96x576xf32>
    %321 = flow.variable.load.indirect %209 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %322 = flow.variable.load.indirect %208 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %323 = flow.variable.load.indirect %207 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %324 = flow.variable.load.indirect %206 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %325 = flow.variable.load.indirect %205 : !iree.ptr<tensor<1x1x576x160xf32>> -> tensor<1x1x576x160xf32>
    %326 = flow.variable.load.indirect %219 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %327 = flow.variable.load.indirect %218 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %328 = flow.variable.load.indirect %217 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %329 = flow.variable.load.indirect %216 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %330 = flow.variable.load.indirect %215 : !iree.ptr<tensor<3x3x960x1xf32>> -> tensor<3x3x960x1xf32>
    %331 = flow.variable.load.indirect %214 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %332 = flow.variable.load.indirect %213 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %333 = flow.variable.load.indirect %212 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %334 = flow.variable.load.indirect %211 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %335 = flow.variable.load.indirect %210 : !iree.ptr<tensor<1x1x160x960xf32>> -> tensor<1x1x160x960xf32>
    %336 = flow.variable.load.indirect %224 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %337 = flow.variable.load.indirect %223 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %338 = flow.variable.load.indirect %222 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %339 = flow.variable.load.indirect %221 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %340 = flow.variable.load.indirect %220 : !iree.ptr<tensor<1x1x960x160xf32>> -> tensor<1x1x960x160xf32>
    %341 = flow.variable.load.indirect %234 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %342 = flow.variable.load.indirect %233 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %343 = flow.variable.load.indirect %232 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %344 = flow.variable.load.indirect %231 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %345 = flow.variable.load.indirect %230 : !iree.ptr<tensor<3x3x960x1xf32>> -> tensor<3x3x960x1xf32>
    %346 = flow.variable.load.indirect %229 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %347 = flow.variable.load.indirect %228 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %348 = flow.variable.load.indirect %227 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %349 = flow.variable.load.indirect %226 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %350 = flow.variable.load.indirect %225 : !iree.ptr<tensor<1x1x160x960xf32>> -> tensor<1x1x160x960xf32>
    %351 = flow.variable.load.indirect %239 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %352 = flow.variable.load.indirect %238 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %353 = flow.variable.load.indirect %237 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %354 = flow.variable.load.indirect %236 : !iree.ptr<tensor<160xf32>> -> tensor<160xf32>
    %355 = flow.variable.load.indirect %235 : !iree.ptr<tensor<1x1x960x160xf32>> -> tensor<1x1x960x160xf32>
    %356 = flow.variable.load.indirect %249 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %357 = flow.variable.load.indirect %248 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %358 = flow.variable.load.indirect %247 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %359 = flow.variable.load.indirect %246 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %360 = flow.variable.load.indirect %245 : !iree.ptr<tensor<3x3x960x1xf32>> -> tensor<3x3x960x1xf32>
    %361 = flow.variable.load.indirect %244 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %362 = flow.variable.load.indirect %243 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %363 = flow.variable.load.indirect %242 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %364 = flow.variable.load.indirect %241 : !iree.ptr<tensor<960xf32>> -> tensor<960xf32>
    %365 = flow.variable.load.indirect %240 : !iree.ptr<tensor<1x1x160x960xf32>> -> tensor<1x1x160x960xf32>
    %366 = flow.variable.load.indirect %254 : !iree.ptr<tensor<320xf32>> -> tensor<320xf32>
    %367 = flow.variable.load.indirect %253 : !iree.ptr<tensor<320xf32>> -> tensor<320xf32>
    %368 = flow.variable.load.indirect %252 : !iree.ptr<tensor<320xf32>> -> tensor<320xf32>
    %369 = flow.variable.load.indirect %251 : !iree.ptr<tensor<320xf32>> -> tensor<320xf32>
    %370 = flow.variable.load.indirect %250 : !iree.ptr<tensor<1x1x960x320xf32>> -> tensor<1x1x960x320xf32>
    %371 = flow.variable.load.indirect %24 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %372 = flow.variable.load.indirect %23 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %373 = flow.variable.load.indirect %22 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %374 = flow.variable.load.indirect %21 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %375 = flow.variable.load.indirect %20 : !iree.ptr<tensor<3x3x96x1xf32>> -> tensor<3x3x96x1xf32>
    %376 = flow.variable.load.indirect %19 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %377 = flow.variable.load.indirect %18 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %378 = flow.variable.load.indirect %17 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %379 = flow.variable.load.indirect %16 : !iree.ptr<tensor<96xf32>> -> tensor<96xf32>
    %380 = flow.variable.load.indirect %15 : !iree.ptr<tensor<1x1x16x96xf32>> -> tensor<1x1x16x96xf32>
    %381 = flow.variable.load.indirect %29 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %382 = flow.variable.load.indirect %28 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %383 = flow.variable.load.indirect %27 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %384 = flow.variable.load.indirect %26 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %385 = flow.variable.load.indirect %25 : !iree.ptr<tensor<1x1x96x24xf32>> -> tensor<1x1x96x24xf32>
    %386 = flow.variable.load.indirect %39 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %387 = flow.variable.load.indirect %38 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %388 = flow.variable.load.indirect %37 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %389 = flow.variable.load.indirect %36 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %390 = flow.variable.load.indirect %35 : !iree.ptr<tensor<3x3x144x1xf32>> -> tensor<3x3x144x1xf32>
    %391 = flow.variable.load.indirect %34 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %392 = flow.variable.load.indirect %33 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %393 = flow.variable.load.indirect %32 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %394 = flow.variable.load.indirect %31 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %395 = flow.variable.load.indirect %30 : !iree.ptr<tensor<1x1x24x144xf32>> -> tensor<1x1x24x144xf32>
    %396 = flow.variable.load.indirect %44 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %397 = flow.variable.load.indirect %43 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %398 = flow.variable.load.indirect %42 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %399 = flow.variable.load.indirect %41 : !iree.ptr<tensor<24xf32>> -> tensor<24xf32>
    %400 = flow.variable.load.indirect %40 : !iree.ptr<tensor<1x1x144x24xf32>> -> tensor<1x1x144x24xf32>
    %401 = flow.variable.load.indirect %54 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %402 = flow.variable.load.indirect %53 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %403 = flow.variable.load.indirect %52 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %404 = flow.variable.load.indirect %51 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %405 = flow.variable.load.indirect %50 : !iree.ptr<tensor<3x3x144x1xf32>> -> tensor<3x3x144x1xf32>
    %406 = flow.variable.load.indirect %49 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %407 = flow.variable.load.indirect %48 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %408 = flow.variable.load.indirect %47 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %409 = flow.variable.load.indirect %46 : !iree.ptr<tensor<144xf32>> -> tensor<144xf32>
    %410 = flow.variable.load.indirect %45 : !iree.ptr<tensor<1x1x24x144xf32>> -> tensor<1x1x24x144xf32>
    %411 = flow.variable.load.indirect %59 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %412 = flow.variable.load.indirect %58 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %413 = flow.variable.load.indirect %57 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %414 = flow.variable.load.indirect %56 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %415 = flow.variable.load.indirect %55 : !iree.ptr<tensor<1x1x144x32xf32>> -> tensor<1x1x144x32xf32>
    %416 = flow.variable.load.indirect %69 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %417 = flow.variable.load.indirect %68 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %418 = flow.variable.load.indirect %67 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %419 = flow.variable.load.indirect %66 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %420 = flow.variable.load.indirect %65 : !iree.ptr<tensor<3x3x192x1xf32>> -> tensor<3x3x192x1xf32>
    %421 = flow.variable.load.indirect %64 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %422 = flow.variable.load.indirect %63 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %423 = flow.variable.load.indirect %62 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %424 = flow.variable.load.indirect %61 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %425 = flow.variable.load.indirect %60 : !iree.ptr<tensor<1x1x32x192xf32>> -> tensor<1x1x32x192xf32>
    %426 = flow.variable.load.indirect %74 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %427 = flow.variable.load.indirect %73 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %428 = flow.variable.load.indirect %72 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %429 = flow.variable.load.indirect %71 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %430 = flow.variable.load.indirect %70 : !iree.ptr<tensor<1x1x192x32xf32>> -> tensor<1x1x192x32xf32>
    %431 = flow.variable.load.indirect %84 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %432 = flow.variable.load.indirect %83 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %433 = flow.variable.load.indirect %82 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %434 = flow.variable.load.indirect %81 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %435 = flow.variable.load.indirect %80 : !iree.ptr<tensor<3x3x192x1xf32>> -> tensor<3x3x192x1xf32>
    %436 = flow.variable.load.indirect %79 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %437 = flow.variable.load.indirect %78 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %438 = flow.variable.load.indirect %77 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %439 = flow.variable.load.indirect %76 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %440 = flow.variable.load.indirect %75 : !iree.ptr<tensor<1x1x32x192xf32>> -> tensor<1x1x32x192xf32>
    %441 = flow.variable.load.indirect %89 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %442 = flow.variable.load.indirect %88 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %443 = flow.variable.load.indirect %87 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %444 = flow.variable.load.indirect %86 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %445 = flow.variable.load.indirect %85 : !iree.ptr<tensor<1x1x192x32xf32>> -> tensor<1x1x192x32xf32>
    %446 = flow.variable.load.indirect %99 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %447 = flow.variable.load.indirect %98 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %448 = flow.variable.load.indirect %97 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %449 = flow.variable.load.indirect %96 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %450 = flow.variable.load.indirect %95 : !iree.ptr<tensor<3x3x192x1xf32>> -> tensor<3x3x192x1xf32>
    %451 = flow.variable.load.indirect %94 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %452 = flow.variable.load.indirect %93 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %453 = flow.variable.load.indirect %92 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %454 = flow.variable.load.indirect %91 : !iree.ptr<tensor<192xf32>> -> tensor<192xf32>
    %455 = flow.variable.load.indirect %90 : !iree.ptr<tensor<1x1x32x192xf32>> -> tensor<1x1x32x192xf32>
    %456 = flow.variable.load.indirect %104 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %457 = flow.variable.load.indirect %103 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %458 = flow.variable.load.indirect %102 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %459 = flow.variable.load.indirect %101 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %460 = flow.variable.load.indirect %100 : !iree.ptr<tensor<1x1x192x64xf32>> -> tensor<1x1x192x64xf32>
    %461 = flow.variable.load.indirect %114 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %462 = flow.variable.load.indirect %113 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %463 = flow.variable.load.indirect %112 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %464 = flow.variable.load.indirect %111 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %465 = flow.variable.load.indirect %110 : !iree.ptr<tensor<3x3x384x1xf32>> -> tensor<3x3x384x1xf32>
    %466 = flow.variable.load.indirect %109 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %467 = flow.variable.load.indirect %108 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %468 = flow.variable.load.indirect %107 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %469 = flow.variable.load.indirect %106 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %470 = flow.variable.load.indirect %105 : !iree.ptr<tensor<1x1x64x384xf32>> -> tensor<1x1x64x384xf32>
    %471 = flow.variable.load.indirect %119 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %472 = flow.variable.load.indirect %118 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %473 = flow.variable.load.indirect %117 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %474 = flow.variable.load.indirect %116 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %475 = flow.variable.load.indirect %115 : !iree.ptr<tensor<1x1x384x64xf32>> -> tensor<1x1x384x64xf32>
    %476 = flow.variable.load.indirect %129 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %477 = flow.variable.load.indirect %128 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %478 = flow.variable.load.indirect %127 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %479 = flow.variable.load.indirect %126 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %480 = flow.variable.load.indirect %125 : !iree.ptr<tensor<3x3x384x1xf32>> -> tensor<3x3x384x1xf32>
    %481 = flow.variable.load.indirect %124 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %482 = flow.variable.load.indirect %123 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %483 = flow.variable.load.indirect %122 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %484 = flow.variable.load.indirect %121 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %485 = flow.variable.load.indirect %120 : !iree.ptr<tensor<1x1x64x384xf32>> -> tensor<1x1x64x384xf32>
    %486 = flow.variable.load.indirect %134 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %487 = flow.variable.load.indirect %133 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %488 = flow.variable.load.indirect %132 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %489 = flow.variable.load.indirect %131 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %490 = flow.variable.load.indirect %130 : !iree.ptr<tensor<1x1x384x64xf32>> -> tensor<1x1x384x64xf32>
    %491 = flow.variable.load.indirect %144 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %492 = flow.variable.load.indirect %143 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %493 = flow.variable.load.indirect %142 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %494 = flow.variable.load.indirect %141 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %495 = flow.variable.load.indirect %140 : !iree.ptr<tensor<3x3x384x1xf32>> -> tensor<3x3x384x1xf32>
    %496 = flow.variable.load.indirect %139 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %497 = flow.variable.load.indirect %138 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %498 = flow.variable.load.indirect %137 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %499 = flow.variable.load.indirect %136 : !iree.ptr<tensor<384xf32>> -> tensor<384xf32>
    %500 = flow.variable.load.indirect %135 : !iree.ptr<tensor<1x1x64x384xf32>> -> tensor<1x1x64x384xf32>
    %501 = flow.variable.load.indirect %149 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %502 = flow.variable.load.indirect %148 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %503 = flow.variable.load.indirect %147 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %504 = flow.variable.load.indirect %146 : !iree.ptr<tensor<64xf32>> -> tensor<64xf32>
    %505 = flow.variable.load.indirect %145 : !iree.ptr<tensor<1x1x384x64xf32>> -> tensor<1x1x384x64xf32>
    %506 = flow.variable.load.indirect %4 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %507 = flow.variable.load.indirect %3 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %508 = flow.variable.load.indirect %2 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %509 = flow.variable.load.indirect %1 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %510 = flow.variable.load.indirect %0 : !iree.ptr<tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
    %511 = flow.variable.load.indirect %259 : !iree.ptr<tensor<1280xf32>> -> tensor<1280xf32>
    %512 = flow.variable.load.indirect %258 : !iree.ptr<tensor<1280xf32>> -> tensor<1280xf32>
    %513 = flow.variable.load.indirect %257 : !iree.ptr<tensor<1280xf32>> -> tensor<1280xf32>
    %514 = flow.variable.load.indirect %256 : !iree.ptr<tensor<1280xf32>> -> tensor<1280xf32>
    %515 = flow.variable.load.indirect %255 : !iree.ptr<tensor<1x1x320x1280xf32>> -> tensor<1x1x320x1280xf32>
    %516 = flow.variable.load.indirect %9 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %517 = flow.variable.load.indirect %8 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %518 = flow.variable.load.indirect %7 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %519 = flow.variable.load.indirect %6 : !iree.ptr<tensor<32xf32>> -> tensor<32xf32>
    %520 = flow.variable.load.indirect %5 : !iree.ptr<tensor<3x3x32x1xf32>> -> tensor<3x3x32x1xf32>
    %521 = flow.variable.load.indirect %14 : !iree.ptr<tensor<16xf32>> -> tensor<16xf32>
    %522 = flow.variable.load.indirect %13 : !iree.ptr<tensor<16xf32>> -> tensor<16xf32>
    %523 = flow.variable.load.indirect %12 : !iree.ptr<tensor<16xf32>> -> tensor<16xf32>
    %524 = flow.variable.load.indirect %11 : !iree.ptr<tensor<16xf32>> -> tensor<16xf32>
    %525 = flow.variable.load.indirect %10 : !iree.ptr<tensor<1x1x32x16xf32>> -> tensor<1x1x32x16xf32>
    %526 = flow.variable.load.indirect %261 : !iree.ptr<tensor<1000xf32>> -> tensor<1000xf32>
    %527 = flow.variable.load.indirect %260 : !iree.ptr<tensor<1280x1000xf32>> -> tensor<1280x1000xf32>
    %528 = "mhlo.convolution"(%arg0, %510) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    %529 = "mhlo.batch_norm_inference"(%528, %509, %508, %507, %506) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %530 = "mhlo.clamp"(%265, %529, %262) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %531 = "mhlo.reshape"(%520) : (tensor<3x3x32x1xf32>) -> tensor<3x3x1x32xf32>
    %532 = "mhlo.convolution"(%530, %531) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 32 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x112x112x32xf32>, tensor<3x3x1x32xf32>) -> tensor<1x112x112x32xf32>
    %533 = "mhlo.batch_norm_inference"(%532, %519, %518, %517, %516) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %534 = "mhlo.clamp"(%265, %533, %262) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %535 = "mhlo.convolution"(%534, %525) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x112x112x32xf32>, tensor<1x1x32x16xf32>) -> tensor<1x112x112x16xf32>
    %536 = "mhlo.batch_norm_inference"(%535, %524, %523, %522, %521) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %537 = "mhlo.convolution"(%536, %380) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x112x112x16xf32>, tensor<1x1x16x96xf32>) -> tensor<1x112x112x96xf32>
    %538 = "mhlo.batch_norm_inference"(%537, %379, %378, %377, %376) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x112x112x96xf32>
    %539 = "mhlo.clamp"(%265, %538, %262) : (tensor<f32>, tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x112x112x96xf32>
    %540 = "mhlo.pad"(%539, %265) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x113x113x96xf32>
    %541 = "mhlo.reshape"(%375) : (tensor<3x3x96x1xf32>) -> tensor<3x3x1x96xf32>
    %542 = "mhlo.convolution"(%540, %541) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 96 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
    %543 = "mhlo.batch_norm_inference"(%542, %374, %373, %372, %371) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %544 = "mhlo.clamp"(%265, %543, %262) : (tensor<f32>, tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x96xf32>
    %545 = "mhlo.convolution"(%544, %385) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x96xf32>, tensor<1x1x96x24xf32>) -> tensor<1x56x56x24xf32>
    %546 = "mhlo.batch_norm_inference"(%545, %384, %383, %382, %381) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %547 = "mhlo.convolution"(%546, %395) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %548 = "mhlo.batch_norm_inference"(%547, %394, %393, %392, %391) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %549 = "mhlo.clamp"(%265, %548, %262) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %550 = "mhlo.reshape"(%390) : (tensor<3x3x144x1xf32>) -> tensor<3x3x1x144xf32>
    %551 = "mhlo.convolution"(%549, %550) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 144 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x56x56x144xf32>
    %552 = "mhlo.batch_norm_inference"(%551, %389, %388, %387, %386) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %553 = "mhlo.clamp"(%265, %552, %262) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %554 = "mhlo.convolution"(%553, %400) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x144xf32>, tensor<1x1x144x24xf32>) -> tensor<1x56x56x24xf32>
    %555 = "mhlo.batch_norm_inference"(%554, %399, %398, %397, %396) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %556 = mhlo.add %546, %555 : tensor<1x56x56x24xf32>
    %557 = "mhlo.convolution"(%556, %410) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %558 = "mhlo.batch_norm_inference"(%557, %409, %408, %407, %406) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %559 = "mhlo.clamp"(%265, %558, %262) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %560 = "mhlo.pad"(%559, %265) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x57x57x144xf32>
    %561 = "mhlo.reshape"(%405) : (tensor<3x3x144x1xf32>) -> tensor<3x3x1x144xf32>
    %562 = "mhlo.convolution"(%560, %561) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 144 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x57x57x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x28x28x144xf32>
    %563 = "mhlo.batch_norm_inference"(%562, %404, %403, %402, %401) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x28x28x144xf32>
    %564 = "mhlo.clamp"(%265, %563, %262) : (tensor<f32>, tensor<1x28x28x144xf32>, tensor<f32>) -> tensor<1x28x28x144xf32>
    %565 = "mhlo.convolution"(%564, %415) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x144xf32>, tensor<1x1x144x32xf32>) -> tensor<1x28x28x32xf32>
    %566 = "mhlo.batch_norm_inference"(%565, %414, %413, %412, %411) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %567 = "mhlo.convolution"(%566, %425) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %568 = "mhlo.batch_norm_inference"(%567, %424, %423, %422, %421) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %569 = "mhlo.clamp"(%265, %568, %262) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %570 = "mhlo.reshape"(%420) : (tensor<3x3x192x1xf32>) -> tensor<3x3x1x192xf32>
    %571 = "mhlo.convolution"(%569, %570) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 192 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %572 = "mhlo.batch_norm_inference"(%571, %419, %418, %417, %416) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %573 = "mhlo.clamp"(%265, %572, %262) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %574 = "mhlo.convolution"(%573, %430) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %575 = "mhlo.batch_norm_inference"(%574, %429, %428, %427, %426) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %576 = mhlo.add %566, %575 : tensor<1x28x28x32xf32>
    %577 = "mhlo.convolution"(%576, %440) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %578 = "mhlo.batch_norm_inference"(%577, %439, %438, %437, %436) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %579 = "mhlo.clamp"(%265, %578, %262) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %580 = "mhlo.reshape"(%435) : (tensor<3x3x192x1xf32>) -> tensor<3x3x1x192xf32>
    %581 = "mhlo.convolution"(%579, %580) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 192 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %582 = "mhlo.batch_norm_inference"(%581, %434, %433, %432, %431) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %583 = "mhlo.clamp"(%265, %582, %262) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %584 = "mhlo.convolution"(%583, %445) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %585 = "mhlo.batch_norm_inference"(%584, %444, %443, %442, %441) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %586 = mhlo.add %576, %585 : tensor<1x28x28x32xf32>
    %587 = "mhlo.convolution"(%586, %455) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %588 = "mhlo.batch_norm_inference"(%587, %454, %453, %452, %451) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %589 = "mhlo.clamp"(%265, %588, %262) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %590 = "mhlo.pad"(%589, %265) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x29x29x192xf32>
    %591 = "mhlo.reshape"(%450) : (tensor<3x3x192x1xf32>) -> tensor<3x3x1x192xf32>
    %592 = "mhlo.convolution"(%590, %591) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 192 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x29x29x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x14x14x192xf32>
    %593 = "mhlo.batch_norm_inference"(%592, %449, %448, %447, %446) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x14x14x192xf32>
    %594 = "mhlo.clamp"(%265, %593, %262) : (tensor<f32>, tensor<1x14x14x192xf32>, tensor<f32>) -> tensor<1x14x14x192xf32>
    %595 = "mhlo.convolution"(%594, %460) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x192xf32>, tensor<1x1x192x64xf32>) -> tensor<1x14x14x64xf32>
    %596 = "mhlo.batch_norm_inference"(%595, %459, %458, %457, %456) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %597 = "mhlo.convolution"(%596, %470) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %598 = "mhlo.batch_norm_inference"(%597, %469, %468, %467, %466) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %599 = "mhlo.clamp"(%265, %598, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %600 = "mhlo.reshape"(%465) : (tensor<3x3x384x1xf32>) -> tensor<3x3x1x384xf32>
    %601 = "mhlo.convolution"(%599, %600) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 384 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %602 = "mhlo.batch_norm_inference"(%601, %464, %463, %462, %461) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %603 = "mhlo.clamp"(%265, %602, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %604 = "mhlo.convolution"(%603, %475) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %605 = "mhlo.batch_norm_inference"(%604, %474, %473, %472, %471) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %606 = mhlo.add %596, %605 : tensor<1x14x14x64xf32>
    %607 = "mhlo.convolution"(%606, %485) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %608 = "mhlo.batch_norm_inference"(%607, %484, %483, %482, %481) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %609 = "mhlo.clamp"(%265, %608, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %610 = "mhlo.reshape"(%480) : (tensor<3x3x384x1xf32>) -> tensor<3x3x1x384xf32>
    %611 = "mhlo.convolution"(%609, %610) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 384 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %612 = "mhlo.batch_norm_inference"(%611, %479, %478, %477, %476) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %613 = "mhlo.clamp"(%265, %612, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %614 = "mhlo.convolution"(%613, %490) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %615 = "mhlo.batch_norm_inference"(%614, %489, %488, %487, %486) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %616 = mhlo.add %606, %615 : tensor<1x14x14x64xf32>
    %617 = "mhlo.convolution"(%616, %500) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %618 = "mhlo.batch_norm_inference"(%617, %499, %498, %497, %496) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %619 = "mhlo.clamp"(%265, %618, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %620 = "mhlo.reshape"(%495) : (tensor<3x3x384x1xf32>) -> tensor<3x3x1x384xf32>
    %621 = "mhlo.convolution"(%619, %620) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 384 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %622 = "mhlo.batch_norm_inference"(%621, %494, %493, %492, %491) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %623 = "mhlo.clamp"(%265, %622, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %624 = "mhlo.convolution"(%623, %505) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %625 = "mhlo.batch_norm_inference"(%624, %504, %503, %502, %501) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %626 = mhlo.add %616, %625 : tensor<1x14x14x64xf32>
    %627 = "mhlo.convolution"(%626, %275) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %628 = "mhlo.batch_norm_inference"(%627, %274, %273, %272, %271) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %629 = "mhlo.clamp"(%265, %628, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %630 = "mhlo.reshape"(%270) : (tensor<3x3x384x1xf32>) -> tensor<3x3x1x384xf32>
    %631 = "mhlo.convolution"(%629, %630) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 384 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %632 = "mhlo.batch_norm_inference"(%631, %269, %268, %267, %266) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %633 = "mhlo.clamp"(%265, %632, %262) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %634 = "mhlo.convolution"(%633, %280) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x384xf32>, tensor<1x1x384x96xf32>) -> tensor<1x14x14x96xf32>
    %635 = "mhlo.batch_norm_inference"(%634, %279, %278, %277, %276) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %636 = "mhlo.convolution"(%635, %290) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %637 = "mhlo.batch_norm_inference"(%636, %289, %288, %287, %286) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %638 = "mhlo.clamp"(%265, %637, %262) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %639 = "mhlo.reshape"(%285) : (tensor<3x3x576x1xf32>) -> tensor<3x3x1x576xf32>
    %640 = "mhlo.convolution"(%638, %639) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 576 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %641 = "mhlo.batch_norm_inference"(%640, %284, %283, %282, %281) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %642 = "mhlo.clamp"(%265, %641, %262) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %643 = "mhlo.convolution"(%642, %295) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %644 = "mhlo.batch_norm_inference"(%643, %294, %293, %292, %291) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %645 = mhlo.add %635, %644 : tensor<1x14x14x96xf32>
    %646 = "mhlo.convolution"(%645, %305) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %647 = "mhlo.batch_norm_inference"(%646, %304, %303, %302, %301) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %648 = "mhlo.clamp"(%265, %647, %262) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %649 = "mhlo.reshape"(%300) : (tensor<3x3x576x1xf32>) -> tensor<3x3x1x576xf32>
    %650 = "mhlo.convolution"(%648, %649) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 576 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %651 = "mhlo.batch_norm_inference"(%650, %299, %298, %297, %296) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %652 = "mhlo.clamp"(%265, %651, %262) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %653 = "mhlo.convolution"(%652, %310) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %654 = "mhlo.batch_norm_inference"(%653, %309, %308, %307, %306) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %655 = mhlo.add %645, %654 : tensor<1x14x14x96xf32>
    %656 = "mhlo.convolution"(%655, %320) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %657 = "mhlo.batch_norm_inference"(%656, %319, %318, %317, %316) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %658 = "mhlo.clamp"(%265, %657, %262) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %659 = "mhlo.pad"(%658, %265) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x15x15x576xf32>
    %660 = "mhlo.reshape"(%315) : (tensor<3x3x576x1xf32>) -> tensor<3x3x1x576xf32>
    %661 = "mhlo.convolution"(%659, %660) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 576 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x15x15x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x7x7x576xf32>
    %662 = "mhlo.batch_norm_inference"(%661, %314, %313, %312, %311) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %663 = "mhlo.clamp"(%265, %662, %262) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %664 = "mhlo.convolution"(%663, %325) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<1x1x576x160xf32>) -> tensor<1x7x7x160xf32>
    %665 = "mhlo.batch_norm_inference"(%664, %324, %323, %322, %321) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %666 = "mhlo.convolution"(%665, %335) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %667 = "mhlo.batch_norm_inference"(%666, %334, %333, %332, %331) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %668 = "mhlo.clamp"(%265, %667, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %669 = "mhlo.reshape"(%330) : (tensor<3x3x960x1xf32>) -> tensor<3x3x1x960xf32>
    %670 = "mhlo.convolution"(%668, %669) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 960 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %671 = "mhlo.batch_norm_inference"(%670, %329, %328, %327, %326) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %672 = "mhlo.clamp"(%265, %671, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %673 = "mhlo.convolution"(%672, %340) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %674 = "mhlo.batch_norm_inference"(%673, %339, %338, %337, %336) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %675 = mhlo.add %665, %674 : tensor<1x7x7x160xf32>
    %676 = "mhlo.convolution"(%675, %350) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %677 = "mhlo.batch_norm_inference"(%676, %349, %348, %347, %346) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %678 = "mhlo.clamp"(%265, %677, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %679 = "mhlo.reshape"(%345) : (tensor<3x3x960x1xf32>) -> tensor<3x3x1x960xf32>
    %680 = "mhlo.convolution"(%678, %679) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 960 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %681 = "mhlo.batch_norm_inference"(%680, %344, %343, %342, %341) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %682 = "mhlo.clamp"(%265, %681, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %683 = "mhlo.convolution"(%682, %355) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %684 = "mhlo.batch_norm_inference"(%683, %354, %353, %352, %351) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %685 = mhlo.add %675, %684 : tensor<1x7x7x160xf32>
    %686 = "mhlo.convolution"(%685, %365) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %687 = "mhlo.batch_norm_inference"(%686, %364, %363, %362, %361) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %688 = "mhlo.clamp"(%265, %687, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %689 = "mhlo.reshape"(%360) : (tensor<3x3x960x1xf32>) -> tensor<3x3x1x960xf32>
    %690 = "mhlo.convolution"(%688, %689) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 960 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %691 = "mhlo.batch_norm_inference"(%690, %359, %358, %357, %356) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %692 = "mhlo.clamp"(%265, %691, %262) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %693 = "mhlo.convolution"(%692, %370) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x960xf32>, tensor<1x1x960x320xf32>) -> tensor<1x7x7x320xf32>
    %694 = "mhlo.batch_norm_inference"(%693, %369, %368, %367, %366) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>) -> tensor<1x7x7x320xf32>
    %695 = "mhlo.convolution"(%694, %515) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x320xf32>, tensor<1x1x320x1280xf32>) -> tensor<1x7x7x1280xf32>
    %696 = "mhlo.batch_norm_inference"(%695, %514, %513, %512, %511) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x7x7x1280xf32>
    %697 = "mhlo.clamp"(%265, %696, %262) : (tensor<f32>, tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x7x7x1280xf32>
    %698 = "mhlo.reduce"(%697, %265) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %710 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%710) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x1280xf32>
    %699 = mhlo.divide %698, %263 : tensor<1x1280xf32>
    %700 = "mhlo.dot"(%699, %527) : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
    %701 = "mhlo.broadcast_in_dim"(%526) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %702 = mhlo.add %700, %701 : tensor<1x1000xf32>
    %703 = "mhlo.reduce"(%702, %264) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %710 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%710) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %704 = "mhlo.broadcast_in_dim"(%703) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %705 = mhlo.subtract %702, %704 : tensor<1x1000xf32>
    %706 = "mhlo.exponential"(%705) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %707 = "mhlo.reduce"(%706, %265) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %710 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%710) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %708 = "mhlo.broadcast_in_dim"(%707) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %709 = mhlo.divide %706, %708 : tensor<1x1000xf32>
    check.expect_almost_eq_const(%709, dense<0.001> : tensor<1x1000xf32>) : tensor<1x1000xf32>
    return
  }
}


#include <benchmark/benchmark.h>
#include <stan/math/rev/mat.hpp>
#include <random>

stan::math::var garch(const std::vector<double>& y, const double sigma1,
		      stan::math::var& mu, stan::math::var& alpha0,
		      stan::math::var& alpha1, stan::math::var& beta1) {
  std::vector<stan::math::var> sigma(y.size());
  sigma[0] = sigma1;
  for (size_t t = 1; t < y.size(); ++t) {
    sigma[t] = stan::math::sqrt(alpha0
				+ alpha1 * stan::math::square(y[t - 1] - mu)
				+ beta1 * stan::math::square(sigma[t - 1]));
  }
  return stan::math::normal_lpdf(y, mu, sigma);
}



static void benchmark_autodiff_stack(benchmark::State& state) {
#ifdef FEATURE_TLS
  stan::math::ChainableStack::init();
#endif
  int T = 200;
  std::vector<double> y
    = {
       4.93766971429527, 4.88991682691714, 5.02546102172474, 4.35567646855897,
       3.83573942983642, 6.42511887092803, 6.21749586660123, 5.3789444976392, 6.12027104532381,
       5.11431135803513, 7.62457311139258, -1.85880465801379, 5.50934094468543, 6.44932572011709,
       5.16341923767196, 3.44639039155509, 4.16984880016615, 4.10647082236182, 4.36148384673768,
       5.95804655550286, 4.04595627245859, 3.12687467791699, 3.00142716630907, 7.2568532076393,
       0.61810605683697, -0.15291709516414, 9.15201288380591, 4.29291689602198, 8.47976545249241,
       3.47121085936776, -0.784460219373412, 6.36436891988627, 7.39246053097208, 7.44821619115044,
       7.94629579597174, 7.45200888445898, 4.91606840711583, 7.07837403999095, 2.27557165708769,
       4.3338510473374, 5.33566695925365, 7.71334572132416, 3.84655561617135, 6.52277390763314,
       3.80731058719347, 5.58548359748507, 4.01715099033084, 3.99054536155583, 5.35642303503983,
       5.63897529833076, 5.88953070348908, 6.0430888347862, 7.01663715231427, 5.23984726391001,
       7.57048294871051, 7.13717882232103, 5.06474214308508, 3.92938942862014, 3.45541765853083,
       4.32754476686183, 8.21224580731755, 5.41823304477533, 4.7841770188398, 3.98404860623278,
       8.26915241265127, 3.33760533950886, 2.06569492404492, 1.52754216877548, 1.83133082640754,
       3.42725863604394, 10.6728548009461, 9.15169891973432, 5.02377347267432, 9.33700652969614,
       6.24136721930321, 6.04950849404453, 5.17506455628691, 3.58392003232125, 2.59548292998048,
       4.83907375200728, 3.9602637043862, 5.82758884414387, 4.23546269160095, 7.22893684131873,
       2.60125320616005, 4.69064165912038, 1.917174792991, 4.61001408936943, 5.47954161943213,
       5.15996686350891, 6.18193831796684, 4.34440919258801, 4.41345809585902, 6.68698472933847,
       3.34899504117051, 6.83270263119169, 4.19524438239594, 6.78734463138665, 3.38096383063052,
       6.91863284632495, 3.68888260517761, 6.26224092273241, 3.44745116922359, 0.562152528549508,
       12.0983927062903, -3.94763062989095, -3.21518975215137, 8.91901621444987, 6.99251510307547,
       8.61130426328963, 0.797295048827984, 0.740760529949786, 6.65043900610575, -1.01025333900225,
       6.01005412829945, 1.05968301738299, 6.82927188819709, 4.16367619052275, 5.12177225953856,
       5.35883603151306, 2.94569636117111, 3.09787782500013, 4.25886372386817, 7.36761963610972,
       2.14698605072961, 7.37538509459182, 4.82724178713162, 4.51204391935278, 5.7304457229641,
       4.41939636949817, 2.75590613231484, 4.36446893309357, 7.16011150309803, 8.29841612795873,
       2.81665431246841, 3.91796707566114, 9.79524802733078, -4.72428858409434, 5.45486794214529,
       6.54469009993541, 6.59733683725192, 6.24159998957624, 3.03968503954618, 1.20935471921342,
       5.26368419728504, 8.64378679332718, 7.49105975619705, 6.47364152057565, 4.52510633927136,
       6.72266533476532, 4.93413298122964, 4.1566114170922, 4.51007640371052, 6.29506991633892,
       3.19826524212404, 5.09675013075576, 3.26616721687184, 5.53757602277581, 6.2441927282187,
       7.20513067270488, 3.07048867275673, 2.74547867330073, 0.981956903350417, 5.28944484748336,
       3.86378897330756, 3.21330962237709, 5.91416547847592, 7.2122398161631, 5.72358999506731,
       6.87125883837987, 2.78265012775101, 3.91399869941797, 5.8714783101321, 4.82252986065352,
       6.44606353404703, 4.90138575295631, 4.76091881679865, 6.56447269598981, 2.61578044200192,
       7.23060033317138, 4.3068921412352, 3.94182008251131, 8.92724502984271, 3.4283380296237,
       1.1672300640445, 0.854351423641126, 12.1460655745991, -6.35075237496737, 7.70559312712892,
       4.51365529175356, 4.9229184146353, 6.46218817415156, 0.285691312540926, 3.64479965114781,
       6.24383143375988, 7.63031493398196, 8.84031816593506, 6.91529144961031, 4.10490141415172,
       5.28480409924716};
  double sigma1 = 0.5;

  std::mt19937 rng(std::random_device{}());

  std::vector<double> gradients(6);
  std::uniform_real_distribution<double> mu_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> zero_one(0, 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(gradients.data());
    stan::math::var mu = mu_dist(rng);
    stan::math::var alpha0 = zero_one(rng);
    stan::math::var alpha1 = zero_one(rng);
    stan::math::var beta1 = zero_one(rng) * (1.0 - alpha1);

    std::vector<stan::math::var> vars = {mu, alpha0, alpha1, beta1};

    stan::math::var lp = garch(y, sigma1, mu, alpha0, alpha1, beta1);
    lp.grad(vars, gradients);
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}


struct coupled_mm_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline std::vector<typename stan::return_type<T1, T2>::type>
  // initial time
  // initial positions
  // parameters
  // double data
  // integer data
  operator()(const T0& t_in, const std::vector<T1>& y,
             const std::vector<T2>& parms, const std::vector<double>& sx,
             const std::vector<int>& sx_int, std::ostream* msgs) const {
    std::vector<typename stan::return_type<T1, T2>::type> ydot(2);

    const T2 act = parms[0];
    const T2 KmA = parms[1];
    const T2 deact = parms[2];
    const T2 KmAp = parms[3];

    ydot[0]
        = -1 * (act * y[0] / (KmA + y[0])) + 1 * (deact * y[1] / (KmAp + y[1]));
    ydot[1]
        = 1 * (act * y[0] / (KmA + y[0])) - 1 * (deact * y[1] / (KmAp + y[1]));

    return (ydot);
  }
};



static void benchmark_autodiff_stack_coupled_mm(benchmark::State& state) {
#ifdef FEATURE_TLS
  stan::math::ChainableStack::init();
#endif
  double t0 = 0;

  std::vector<double> ts_long;
  ts_long.push_back(1E3);

  std::vector<double> ts_short;
  ts_short.push_back(1);

  std::vector<double> data;
  std::vector<int> data_int;

  std::vector<double> gradients(6);
  coupled_mm_ode_fun f_;

  for (auto _ : state) {
    benchmark::DoNotOptimize(gradients.data());
    std::vector<stan::math::var> theta
      = {0.932858, 1.27742, 5.40574, 0.1821505};
    std::vector<stan::math::var> y0_v
      = {158.981, 20.7287};

    std::vector<stan::math::var> vars
      = {theta[0], theta[1], theta[2], theta[3], y0_v[0], y0_v[1]};

    std::vector<std::vector<stan::math::var>> res
      = stan::math::integrate_ode_rk45(f_, y0_v, t0, ts_long, theta, data,
				       data_int, 0, 1E-6, 1E-6, 1000000000);

    res[0][0].grad(vars, gradients);
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

static void benchmark_autodiff_stack_coupled_mm_nested(benchmark::State& state) {
#ifdef FEATURE_TLS
  stan::math::ChainableStack::init();
#endif
  double t0 = 0;

  std::vector<double> ts_long;
  ts_long.push_back(1E3);

  std::vector<double> ts_short;
  ts_short.push_back(1);

  std::vector<double> data;
  std::vector<int> data_int;

  std::vector<double> gradients(6);
  coupled_mm_ode_fun f_;

  for (auto _ : state) {
    benchmark::DoNotOptimize(gradients.data());
    for (int n = 0; n < 2; ++n) {
      stan::math::start_nested();

      std::vector<stan::math::var> theta
	= {0.932858, 1.27742, 5.40574, 0.1821505};
      std::vector<stan::math::var> y0_v
	= {158.981, 20.7287};

      std::vector<stan::math::var> vars
	= {theta[0], theta[1], theta[2], theta[3], y0_v[0], y0_v[1]};

      std::vector<std::vector<stan::math::var>> res
	= stan::math::integrate_ode_rk45(f_, y0_v, t0, ts_long, theta, data,
					 data_int, 0, 1E-6, 1E-6, 1000000000);

      res[0][n].grad(vars, gradients);
      stan::math::recover_memory_nested();
      benchmark::ClobberMemory();
    }
    stan::math::recover_memory();
  }
}


BENCHMARK(benchmark_autodiff_stack);
//BENCHMARK(benchmark_autodiff_stack_coupled_mm);
//BENCHMARK(benchmark_autodiff_stack_coupled_mm_nested);

BENCHMARK_MAIN();

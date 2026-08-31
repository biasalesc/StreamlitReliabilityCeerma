from reliability.Distributions import Weibull_Distribution, \
    Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, \
    Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, \
    Gumbel_Distribution
from reliability.Fitters import Fit_Exponential_1P, Fit_Exponential_2P, \
    Fit_Lognormal_2P, Fit_Lognormal_3P, Fit_Normal_2P, Fit_Weibull_2P, \
    Fit_Weibull_3P, Fit_Gamma_2P, Fit_Gamma_3P, Fit_Gumbel_2P, \
    Fit_Loglogistic_2P, Fit_Loglogistic_3P, Fit_Beta_2P
from reliability.Nonparametric import KaplanMeier, NelsonAalen, RankAdjustment
from reliability.ALT_fitters import Fit_Weibull_Exponential, \
    Fit_Weibull_Eyring, Fit_Weibull_Power, Fit_Weibull_Dual_Exponential, \
    Fit_Weibull_Power_Exponential, Fit_Weibull_Dual_Power, \
    Fit_Lognormal_Exponential, Fit_Lognormal_Eyring, \
    Fit_Lognormal_Power, Fit_Lognormal_Dual_Exponential, \
    Fit_Lognormal_Power_Exponential, Fit_Lognormal_Dual_Power, \
    Fit_Normal_Exponential, Fit_Normal_Eyring, Fit_Normal_Power, \
    Fit_Normal_Dual_Exponential, Fit_Normal_Power_Exponential, \
    Fit_Normal_Dual_Power, Fit_Exponential_Exponential, \
    Fit_Exponential_Eyring, Fit_Exponential_Power, \
    Fit_Exponential_Dual_Exponential, Fit_Exponential_Power_Exponential, \
    Fit_Exponential_Dual_Power


exponential_equations = [
    r'\lambda = \text{Scale parameter } (\lambda > 0 )',
    r'\text{Limits: } (t \geq 0) ',
    r'\text{PDF: } f(t) = \lambda e^{-\lambda t}',
    r'\text{CDF: } F(t) = 1 - \lambda e^{-\lambda t}',
    r'R(t) =  e^{-\lambda t}',
    r'\text{HF: } h(t) = \lambda',
    r'\text{CHF: } H(t) = \lambda t',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]
weibull_equations = [
    r'\alpha = \text{Scale parameter } (\alpha > 0 )',
    r'\beta = \text{Shape parameter } (\beta > 0 )',
    r'\text{Limits: } (t \leq 0) ',
    r'\text{PDF: } f(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{(\beta-1)}e^{-(\frac{t}{\alpha})^\beta}',
    r'\text{CDF: } F(t) = 1-e^{-(\frac{t}{\alpha})^\beta}',
    r'R(t) =  e^{-(\frac{t}{\alpha})^\beta}',
    r'\text{HF: } h(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{(\beta-1)}',
    r'\text{CHF: } H(t) = (\frac{t}{\alpha})^\beta',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]
normal_equations = [
    r'\mu = \text{Location parameter } (-\infty<\mu<\infty)',
    r'\sigma = \text{Scale parameter } (\sigma>0)',
    r'\text{Limits: } (-\infty<t<\infty) ',
    r'\text{PDF: } f(t) = \frac{1}{\sigma\sqrt{2\pi}} e^{\frac{1}{2}\left(\frac{t-\mu}{\sigma}\right)^2}=\frac{1}{\sigma}\phi\left[\frac{t-\mu}{\sigma}\right]',
    r'\text{Where } \phi\text{ is the standard normal PDF with }\mu=0\text{ and }\sigma=1',
    r'\text{CDF: } F(t) = \frac{1}{\sigma\sqrt{2\pi}}\int^t_{-\infty}e^{\left[-\frac{1}{2}\left(\frac{\theta-\mu}{\sigma}\right)^2\right]d\theta}=\Phi\left(\frac{t-\mu}{\sigma}\right)',
    r'\text{Where } \Phi\text{ is the standard normal CDF with }\mu=0\text{ and }\sigma=1',
    r'R(t) =  1-\Phi\left(\frac{t-\mu}{\sigma}\right)=\Phi\left(\frac{\mu-t}{\sigma}\right)',
    r'\text{HF: } h(t) = \frac{\phi\left[\frac{t-\mu}{\sigma}\right]}{\sigma\left(\Phi\left[\frac{\mu-t}{\sigma}\right]\right)}',
    r'\text{CHF: } H(t) = -ln\left[\Phi\left(\frac{\mu-t}{\sigma}\right)\right]',
]
lognormal_equations = [
    r'\alpha = \text{Scale parameter } (-\infty<\mu<\infty)',
    r'\beta = \text{Shape parameter } (\sigma>0)',
    r'\text{Limits: } (t \leq 0)',
    r'\text{PDF: } f(t) = \frac{1}{\sigma t\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{ln(t)-\mu}{\sigma}\right)^2}=\frac{1}{\sigma t}\phi\left[\frac{ln(t)-\mu}{\sigma}\right]',
    r'\text{Where } \phi \text{ is the standard normal PDF with }\mu=0\text{ and }\sigma=1 ',
    r'\text{CDF: } F(t) = \frac{1}{\sigma\sqrt{2\pi}}\int^t_{0}\frac{1}{\theta}e^{\left[-\frac{1}{2}\left(\frac{ln(\theta)-\mu}{\sigma}\right)^2\right]d\theta}=\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)',
    r'\text{Where } \Phi \text{ is the standard normal CDF with }\mu=0\text{ and }\sigma=1',
    r'R(t) =  1-\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)=\Phi\left(\frac{\mu-ln(t)}{\sigma}\right)',
    r'\text{HF: } h(t) = \frac{\phi\left[\frac{ln(t)-\mu}{\sigma}\right]}{\sigma\left(\Phi\left[\frac{\mu-ln(t)}{\sigma}\right]\right)}',
    r'\text{CHF: } H(t) = -ln\left[1-\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)\right]',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]
beta_equations = [
    r'\alpha = \text{Shape parameter } (\alpha>0)',
    r'\beta = \text{Shape parameter } (\beta>0)',
    r'\text{Limits: } (0 \leq t \leq1)',
    r'\text{PDF: } f(t) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}t^{\alpha-1}(1-t)^{\beta-1}=\frac{t^{\alpha -1}(1-t)^{\beta-1}}{B(\alpha,\beta)}',
    r'\text{Where } \Gamma(x) \text{ is the complete gamma function. } \Gamma(x)=\int^\infty_{0}t^{x-1}e^{-t}dt',
    r'\text{Where } B(x,y) \text{ is the complete beta function. } B(x,y)=\int^{1}_{0}(1- t)^{y-1}dt',
    r'\text{CDF: } F(t) = \frac{\Gamma(\alpha +\beta)}{\Gamma(\alpha)\Gamma(\beta)}\int^{t}_{0}\theta^{\alpha-1}(1-\theta^{\beta-1}d\theta=\frac{B_t(t|\alpha,\beta)}{B(\alpha,\beta)}=I_t(t|\alpha,\beta)',
    r'\text{Where } B_t(t|x,y) \text{  is the incomplete beta function.} B_t(t|x,y)=\int^{t}_{0}\theta^{x-1}(1-\theta)^{y-1}d\theta',
    r'\text{Where } I_t(t|x,y) \text{ is the regularized incomplete beta function which is defined in terms}',
    r'\text{ of the incomplete beta function and the complete beta function.} I_t(t|x,y)=\frac{B_t(t|x,y)}{B_t(x,y)}',
    r'R(t) =  1-I_t(t|\alpha,\beta)',
    r'\text{HF: } h(t) = \frac{t^{y-1}(1-t)}{B(\alpha,\beta)-B_t(t|\alpha,\beta)}',
    r'\text{CHF: } H(t) = -ln\left[1-I_t(t|\alpha,\beta)\right]',
]
gamma_equations = [
    r'\alpha = \text{Scale parameter } ( \alpha > 0)',
    r'\beta = \text{Shape parameter } ( \beta > 0)',
    r'\text{Limits: } ( t \leq 0 ) ',
    r'\text{PDF: } f(t) = \frac{t^{\beta-1}}{\Gamma(\beta)\alpha^\beta}e^{\frac{t}{\alpha}}',
    r'\text{Where } \Gamma(x)\text{ is the complete gamma function. }\Gamma(x)=\int^\infty_{0}t^{x-1}e^{-t} dt',
    r'\text{CDF: } F(t) = \frac{1}{\Gamma(\beta)}\gamma(\beta,\frac{t}{\alpha})',
    r'\text{Where } \gamma(x,y) \text{ is the lower incomplete gamma function. }\gamma(x,y)=\frac{1}{\Gamma(x)}\int^y_{0}t^{x-1}e^{-t}dt',
    r'R(t) =  \frac{\Gamma(\beta,\frac{t}{\alpha})}{\Gamma(\beta)}',
    r'\text{HF: } h(t) = \frac{t^{\beta-1}e^{-\frac{t}{\alpha}}}{\alpha^\beta\Gamma(\beta,\frac{t}{\alpha})}',
    r'\text{CHF: } H(t) = -ln\left[\frac{1}{\Gamma(\beta)}\Gamma(\beta,\frac{t}{\alpha})\right]',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]
gumbel_equations = [
    r'\mu = \text{Location parameter } (-\infty<\mu<\infty)',
    r'\sigma = \text{Scale parameter } (\sigma>0)',
    r'\text{Limits: } (-\infty<t<\infty)',
    r'\text{PDF: } f(t) = \frac{1}{\sigma}e^{z-e^{z}}',
    r'\text{Where } z = \frac{t-\mu}{\sigma}',
    r'\text{CDF: } F(t) = 1-e^{-e^{z}}',
    r'R(t) =  e^{-e^{z}}',
    r'\text{HF: } h(t) = \frac{e^{z}}{\sigma}',
    r'\text{CHF: } H(t) = e^{z}',
]
loglogistic_equations = [
    r'\alpha = \text{Scale parameter } ( \alpha > 0)',
    r'\beta = \text{Shape parameter } ( \beta > 0)',
    r'\text{Limits: } ( t \leq 0 )',
    r'\text{PDF: } f(t) = \frac{\frac{\beta}{\alpha}(\frac{t}{\alpha})^{\beta-1}}{\left(1+(\frac{t}{\alpha})^\beta\right)^2}',
    r'\text{CDF: } F(t) = \frac{1}{1+(\frac{t}{\alpha})^{-\beta}}=\frac{(\frac{t}{\alpha})^{\beta}}{1+(\frac{t}{\alpha})^{\beta}}=\frac{t^{\beta}}{\alpha^{\beta}+t^{\beta}}',
    r'R(t) =  \frac{1}{1+(\frac{t}{\alpha})^{\beta}}',
    r'\text{HF: } h(t) = \frac{\frac{\beta}{\alpha}(\frac{t}{\alpha})^{\beta-1}}{1+(\frac{t}{\alpha})^\beta}',
    r'\text{CHF: } H(t) = -ln\left(1+(\frac{t}{\alpha})^{\beta}\right)',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]


exponential_info = {
    'n_params': 2,
    'equations': exponential_equations,
    'distribution': Exponential_Distribution,
    'variables': {'Rate parameter λ': [10.0, 0.0],
                  'Location parameter γ': [0.0, 0.0]},
}
weibull_info = {
    'n_params': 3,
    'equations': weibull_equations,
    'distribution': Weibull_Distribution,
    'variables': {'Scale parameter α': [10.0, 0.0],
                  'Shape parameter β': [1.0, 0.0],
                  'Location parameter γ': [0.0, 0.0]},
}
normal_info = {
    'n_params': 2,
    'equations': normal_equations,
    'distribution': Normal_Distribution,
    'variables': {'Mean parameter μ': [0.0, None],
                  'Standard deviation parameter σ': [1.0, 0.0]},
}
lognormal_info = {
    'n_params': 3,
    'equations': lognormal_equations,
    'distribution': Lognormal_Distribution,
    'variables': {'Mean parameter (of log-time) μ': [0.0, None],
                  'Standard deviation parameter (of log-time) σ': [1.0, 0.0],
                  'Location parameter γ': [0.0, 0.0]},
}
beta_info = {
    'n_params': 2,
    'equations': beta_equations,
    'distribution': Beta_Distribution,
    'variables': {'Shape parameter α': [1.0, 0.0],
                  'Shape parameter β': [1.0, 0.0]},
}
gamma_info = {
    'n_params': 3,
    'equations': gamma_equations,
    'distribution': Gamma_Distribution,
    'variables': {'Scale parameter α': [1.0, 0.0],
                  'Shape parameter β': [1.0, 0.0],
                  'Location parameter γ': [0.0, 0.0]},
}
gumbel_info = {
    'n_params': 2,
    'equations': gumbel_equations,
    'distribution': Gumbel_Distribution,
    'variables': {'Location parameter μ': [0.0, None],
                  'Scale parameter σ': [1.0, 0.0]},
}
loglogistic_info = {
    'n_params': 3,
    'equations': loglogistic_equations,
    'distribution': Loglogistic_Distribution,
    'variables': {'Scale parameter α': [1.0, 0.0],
                  'Shape parameter β': [5.0, 0.0],
                  'Location parameter γ': [0.0, 0.0]},
}


distributions = {
    'Exponential Distribution': exponential_info,
    'Weibull Distribution': weibull_info,
    'Normal Distribution': normal_info,
    'Lognormal Distribution': lognormal_info,
    'Beta Distribution': beta_info,
    'Gamma Distribution': gamma_info,
    'Gumbel Distribution': gumbel_info,
    'Loglogistic Distribution': loglogistic_info,
}


fit_distributions = {
    'Exponential_1P': [Fit_Exponential_1P, exponential_info],
    'Exponential_2P': [Fit_Exponential_2P, exponential_info],
    'Weibull_2P': [Fit_Weibull_2P, weibull_info],
    'Weibull_3P': [Fit_Weibull_3P, weibull_info],
    'Normal_2P': [Fit_Normal_2P, normal_info],
    'Lognormal_2P': [Fit_Lognormal_2P, lognormal_info],
    'Lognormal_3P': [Fit_Lognormal_3P, lognormal_info],
    'Beta_2P': [Fit_Beta_2P, beta_info],
    'Gamma_2P': [Fit_Gamma_2P, gamma_info],
    'Gamma_3P': [Fit_Gamma_3P, gamma_info],
    'Gumbel_2P': [Fit_Gumbel_2P, gumbel_info],
    'Loglogistic_2P': [Fit_Loglogistic_2P, loglogistic_info],
    'Loglogistic_3P': [Fit_Loglogistic_3P, loglogistic_info],
}

# Weibull_Mixture, Weibull_CR and Weibull_DS are composite models (they combine
# multiple sub-distributions) that reliability.Fitters.Fit_Everything fits by
# default. They don't fit the "one Fit_X function + one Parameter table" shape
# assumed by fit_distributions above, so show_fitter.py cannot build the
# best-fit follow-up (probability plot / CI table) for them and must always
# exclude them from Fit_Everything to avoid a KeyError when one of them is
# selected as the best fit.
unsupported_distributions = ['Weibull_Mixture', 'Weibull_CR', 'Weibull_DS']


non_parametric_distributions = {
    'Kaplan-Meier': KaplanMeier,
    'Nelson-Aalen': NelsonAalen,
    'Rank Adjustment': RankAdjustment,
}


alt_single_distributions = {
    'Weibull_Exponential': Fit_Weibull_Exponential,
    'Weibull_Eyring': Fit_Weibull_Eyring,
    'Weibull_Power': Fit_Weibull_Power,
    'Lognormal_Exponential': Fit_Lognormal_Exponential,
    'Lognormal_Eyring': Fit_Lognormal_Eyring,
    'Lognormal_Power': Fit_Lognormal_Power,
    'Normal_Exponential': Fit_Normal_Exponential,
    'Normal_Eyring': Fit_Normal_Eyring,
    'Normal_Power': Fit_Normal_Power,
    'Exponential_Exponential': Fit_Exponential_Exponential,
    'Exponential_Eyring': Fit_Exponential_Eyring,
    'Exponential_Power': Fit_Exponential_Power,
}


alt_dual_distributions = {
    'Weibull_Dual_Exponential': Fit_Weibull_Dual_Exponential,
    'Weibull_Power_Exponential': Fit_Weibull_Power_Exponential,
    'Weibull_Dual_Power': Fit_Weibull_Dual_Power,
    'Lognormal_Dual_Exponential': Fit_Lognormal_Dual_Exponential,
    'Lognormal_Power_Exponential': Fit_Lognormal_Power_Exponential,
    'Lognormal_Dual_Power': Fit_Lognormal_Dual_Power,
    'Normal_Dual_Exponential': Fit_Normal_Dual_Exponential,
    'Normal_Power_Exponential': Fit_Normal_Power_Exponential,
    'Normal_Dual_Power': Fit_Normal_Dual_Power,
    'Exponential_Dual_Exponential': Fit_Exponential_Dual_Exponential,
    'Exponential_Power_Exponential': Fit_Exponential_Power_Exponential,
    'Exponential_Dual_Power': Fit_Exponential_Dual_Power,
}


alt_substitution_equations = [
    r'\text{Exponential } \lambda = \frac{1}{L(S)}',
    r'\text{Weibull } \alpha = L(S)',
    r'\text{Normal } \mu = L(S)',
    r'\text{Lognormal } \mu = \ln\left(L(S)\right)',
]
alt_single_equations = [
    r'\text{Exponential (or Arrhenius): } L(S)=b*\exp\left(\frac{a}{S}\right)',
    r'\text{Eyring: } L(S)=\frac{1}{S}*\exp\left(-\left(c-\frac{a}{S}\right)\right)',
    r'\text{Power (or Inverse Power Law): } L(S)=a*S^n',
]
alt_dual_equations = [
    r'\text{Dual-Exponential (or Temperature-Humidity): } L(S_1,S_2)=c*\exp\left(\frac{a}{S_1}+\frac{b}{S_2}\right)',
    r'\text{Dual-Power: } L(S_1,S_2)=c*S_1^m*S_2^n',
    r'\text{Power-Exponential (or Thermal-Nonthermal): } L(S_1,S_2)=c*\exp\left(\frac{a}{S_1}\right)*S_2^n',
]
acceleration_factor_equations = [
    r'\text{Arrhenius (or Exponential): } A_F = \exp\left({\frac{B}{S_{use}} - \frac{B}{S_{acc}}}\right)',
    r'\text{Eyring: } A_F = \frac{S_{acc}}{S_{use}}\exp\left({\frac{B}{S_{use}} - \frac{B}{S_{acc}}}\right)',
    r'\text{Inverse Power Law (or Power): } A_F = \left(\frac{S_{acc}}{S_{use}}\right)^B',
]

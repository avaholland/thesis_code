import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import thejoker as tj
from astropy.time import Time
from astropy.visualization.units import quantity_support


rnd = np.random.default_rng(seed = 42)
data_tbl = at.QTable.read("Douglas_ComaBer_match.tsv", format="ascii.tab")
#print(data_tbl.colnames)
print(len(data_tbl))

unique_names1 = set(data_tbl["name"])
print(len(unique_names1))
#print(data_tbl["name", "RV"][:20])


data_tbl2 = at.QTable.read("dbdump_comaber.csv")
unique_names2 = set(data_tbl2["Candidates.CandName"])
print(len(unique_names2))
#print(data_tbl2["Candidates.CandName", "Exposures.RV"][1000:1020])
print(len(data_tbl2))
#print(data_tbl[:2])
print(data_tbl2.colnames)

mask2 = [str(name).startswith('CB') for name in data_tbl2["Candidates.CandName"]]

cb_candidates2 = data_tbl2[mask2]
#print(len(cb_candidates2))
#print(cb_candidates2["Candidates.CandName", "Exposures.RV"][9:48]) #860 rows start w CB, rows 9-48 are CB0002


mask = [str(name).startswith('CB') for name in data_tbl["name"]]
cb_candidates = data_tbl[mask]
#print(cb_candidates["name", "RV"][1106:]) #1149 rows start with CB, rows 1106 - 1149 are CB0002

mask_table = [str(name).startswith('CB0002') for name in data_tbl["name"]]
cb0002 = data_tbl[mask_table]
cb0002.sort("RV")
#print(cb0002["name", "RV"])

mask_table2 = [str(name).startswith('CB0002') for name in data_tbl2["Candidates.CandName"]]
cb0002_2 = data_tbl2[mask_table2]
cb0002_2.sort("Exposures.RV")
#print(cb0002_2["Candidates.CandName", "Exposures.RV"])

#print(len(cb_candidates))
#sub_tbl = data_tbl[rnd.choice(len(data_tbl), size=4, replace=False)]
#t = Time(sub_tbl["JD"], format="jd", scale="tcb")
#data = tj.RVData(t=t, rv=sub_tbl["rv"])

sub_tbl2 = data_tbl2[rnd.choice(len(data_tbl2), size=4, replace=False)]

"""
# Convert time for ALL exposures of CB0002
t = Time(cb0002_2["Exposures.JD"], format="jd", scale="utc")

# Create RVData using ALL exposures of CB0002
data2 = tj.RVData(
    t=t, 
    rv=cb0002_2["Exposures.RV"] * (u.km / u.s), 
    rv_err=cb0002_2["Exposures.RV_err"] * (u.km / u.s)
)



prior = tj.JokerPrior.default(
    P_min=1e-1 * u.day,
    P_max=5e3 * u.day,
    sigma_K0=30 * u.km / u.s,
    sigma_v=100 * u.km / u.s,
)
prior_samples = prior.sample(size=250_000, rng=rnd)

joker = tj.TheJoker(prior, rng=rnd)
joker_samples = joker.rejection_sample(data2, prior_samples, max_posterior_samples=256)
_ = tj.plot_rv_curves(joker_samples, data=data2)
plt.show()
"""

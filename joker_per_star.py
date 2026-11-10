import os
import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import thejoker as tj
from astropy.time import Time
from astropy.visualization.units import quantity_support
from astropy.coordinates import SkyCoord
import corner

output_dir = "/Users/avaholland/thesis/plots"

rnd = np.random.default_rng(seed = 42)
data_dump = at.QTable.read("dbdump_comaber.csv")

#filtering out non-comaber stars based on coordinates
ra_col = "Candidates.RA"
dec_col = "Candidates.Dec"

#SkyCoord for every exposure
all_coords = SkyCoord(
    ra = np.asarray(data_dump["Candidates.RA"], dtype=float) * u.deg,
    dec = np.asarray(data_dump["Candidates.Dec"], dtype=float) * u.deg,
    frame = "icrs"
)

#Keep only the exposures within 20 degrees of comaber, is this the right coordinate for comaber?
comaber = SkyCoord(ra = 186.3 * u.deg, dec = 26.1 * u.deg, frame = "icrs")
close_exposures = data_dump[all_coords.separation(comaber) < 20 * u.deg]

#Unique names among those exposures only
close_names = [str(name) for name in np.unique(close_exposures["Candidates.CandName"])]
#print(len(close_names), "unique stars within 20 degrees")
#print(close_names[82])
#print(close_names[:3])

#Loop over those names
def run_joker(star_name, exposures, rnd, n_prior_samples=250_000):
    data_tbl = exposures[exposures["Candidates.CandName"] == star_name]
    t = Time(data_tbl["Exposures.JD"], format="jd", scale="utc")
    data = tj.RVData(
        t = t,
        rv = data_tbl["Exposures.RV"] * (u.km / u.s),
        rv_err = data_tbl["Exposures.RV_err"] * (u.km / u.s)
        )
    prior = tj.JokerPrior.default(
        P_min = 1e-1 * u.day,
        P_max = 5e3 * u.day,
        sigma_K0 = 30 * u.km / u.s,
        sigma_v = 100 * u.km / u.s,
        )
    prior_samples = prior.sample(size=n_prior_samples, rng=rnd)
    joker = tj.TheJoker(prior, rng=rnd)
    joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)
    return data, joker_samples

def plot_joker(star_name, data, joker_samples, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    _ = tj.plot_rv_curves(joker_samples, data=data, ax=ax)
    ax.set_title(f"Radial Velocity Fit for {star_name}")
    save_path = os.path.join(output_dir, f"{star_name}_rv_fit.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return data, joker_samples

def corner_plot(star_name, data, samples, output_dir):
    params = np.column_stack([
        samples["P"].to_value(u.day),
        samples["e"],
        samples["K"].to_value(u.km / u.s),
        samples["v0"].to_value(u.km / u.s),
        ])
    fig = corner.corner(
        params,
        labels = ["p", "e", "K", "v0"],
        plot_datapoints = True,
        plot_density = False,
        data_kwargs = {"alpha": 0.1, "ms": 2},
        color = "k",
        fig = plt.figure(figsize = (10, 10)),
        )
    axes = np.array(fig.axes).reshape(4, 4)
    ax_rv = axes[3, 3]
    ax_rv.clear()
    tj.plot_rv_curves(samples, data = data, ax = ax_rv)
    save_path = os.path.join(output_dir, f"{star_name}_corner.png")
    fig.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

for star_name in close_names[1:5]:
    data, joker_samples = run_joker(star_name, close_exposures, rnd)
    n_samples = len(joker_samples)
    plot_joker(star_name, data, joker_samples, output_dir)
    if n_samples < 5:
        print("   too few samples for a corner plot, skipping it", flush=True)
        continue
    corner_plot(star_name, data, joker_samples, output_dir)

"""

for star_name in close_names[80:85]:
    data_tbl = close_exposures[close_exposures["Candidates.CandName"] == star_name]
    t = Time(data_tbl["Exposures.JD"], format="jd", scale="utc")
    data = tj.RVData(
        t = t,
        rv = data_tbl["Exposures.RV"] * (u.km / u.s),
        rv_err = data_tbl["Exposures.RV_err"] * (u.km / u.s)
    )
    prior = tj.JokerPrior.default(
        P_min = 1e-1 * u.day,
        P_max = 5e3 * u.day,
        sigma_K0 = 30 * u.km / u.s,
        sigma_v = 100 * u.km / u.s,
    )
    prior_samples = prior.sample(size=250_000, rng=rnd)
    joker = tj.TheJoker(prior, rng=rnd)
    joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

    fig, ax = plt.subplots(figsize=(10, 6))
    _ = tj.plot_rv_curves(joker_samples, data=data, ax=ax)
    ax.set_title(f"Radial Velocity Fit for {star_name}")
    save_path = os.path.join(output_dir, f"{star_name}_rv_fit.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
"""
"""
comaber = SkyCoord(ra = 186.3 * u.deg, dec = 26.1 * u.deg, frame = "icrs")
separation = star_coords.separation(comaber)
within_20 = separation < 20 * u.deg
#print(len(stars), "stars total")
#print(within_20.sum(), "stars within 20 degrees of comaber")
#print("Stars within 20 degrees:")
#print(stars["Candidates.CandName"][within_20])



for star_name in close_names[:3]:
	mask = [str(name).startswith(star_name) for name in data_dump["Candidates.CandName"]]
	data_tbl = data_dump[mask]
	t = Time(data_tbl["Exposures.JD"], format="jd", scale="utc")
	data = tj.RVData(
    	t = t, 
    	rv = data_tbl["Exposures.RV"] * (u.km / u.s), 
    	rv_err = data_tbl["Exposures.RV_err"] * (u.km / u.s)
	)
	prior = tj.JokerPrior.default(
   		P_min = 1e-1 * u.day,
    	P_max = 5e3 * u.day,
    	sigma_K0 = 30 * u.km / u.s,
    	sigma_v = 100 * u.km / u.s,
	)	
	prior_samples = prior.sample(size=250_000, rng=rnd)
	joker = tj.TheJoker(prior, rng=rnd)
	joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

	fig, ax = plt.subplots(figsize=(10, 6))
	_ = tj.plot_rv_curves(joker_samples, data=data, ax=ax)
	ax.set_title(f"Radial Velocity Fit for {star_name}")
	save_path = os.path.join(output_dir, f"{star_name}_rv_fit.png")
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close(fig)

star_name = "CB0002"
#change this all basaed on star name
mask = [str(name).startswith(star_name) for name in data_dump["Candidates.CandName"]]
cb0002_2 = data_dump[mask]

t = Time(cb0002_2["Exposures.JD"], format="jd", scale="utc")
data = tj.RVData(
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
joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

fig, ax = plt.subplots(figsize=(10, 6))
_ = tj.plot_rv_curves(joker_samples, data=data, ax=ax)
ax.set_title(f"Radial Velocity Fit for {star_name}")
save_path = os.path.join(output_dir, f"{star_name}_rv_fit.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)


"""

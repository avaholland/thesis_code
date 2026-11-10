import os
import astropy.table as at
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
import thejoker as tj
from astropy.time import Time
from astropy.visualization.units import quantity_support
from astropy.coordinates import SkyCoord
import corner
import pymc as pm
import pickle


output_dir = "/Users/avaholland/thesis/plots"

rnd = np.random.default_rng(seed = 42)

data_dump = at.QTable.read("dbdump_comaber.csv")
ra_col = "Candidates.RA"
dec_col = "Candidates.Dec"

all_coords = SkyCoord(
    ra = np.asarray(data_dump["Candidates.RA"], dtype=float) * u.deg,
    dec = np.asarray(data_dump["Candidates.Dec"], dtype=float) * u.deg,
    frame = "icrs"
)

comaber = SkyCoord(ra = 186.3 * u.deg, dec = 26.1 * u.deg, frame = "icrs")
close_exposures = data_dump[all_coords.separation(comaber) < 20 * u.deg]

close_names = [str(name) for name in np.unique(close_exposures["Candidates.CandName"])]

keeping_stars = []

for name in close_names:
	n_exposures = (data_dump["Candidates.CandName"] == name).sum()
	if n_exposures > 2:
		keeping_stars.append(name)

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
	return data, joker_samples, prior, joker

def run_mcmc(data, joker_samples, prior, joker, mcmc_file):
	with prior.model:
		mcmc_init = joker.setup_mcmc(data, joker_samples)
		trace = pm.sample(tune = 500, draws = 500, start = mcmc_init, cores = 1, chains = 2, init = "adapt_full")
	mcmc_samples = tj.JokerSamples.from_inference_data(prior, trace, data)
	mcmc_samples.wrap_K()
	mcmc_samples.write(mcmc_file, overwrite = True)
	return mcmc_samples

def plot_joker(star_name, data, joker_samples, output_dir):
	fig, ax = plt.subplots(figsize = (10, 6))
	_ = tj.plot_rv_curves(joker_samples, data = data, ax = ax)
	ax.set_title(f"Raidal Velocity Fit for {star_name}")
	save_path = os.path.join(output_dir, f"{star_name}_rv_fit.png")
	plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
	plt.close(fig)
	

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

for star_name in keeping_stars[:1]:
	data, joker_samples, prior, joker = run_joker(star_name, close_exposures, rnd)
	n_samples = len(joker_samples)
	plot_joker(star_name, data, joker_samples, output_dir)
	plt.close("all")
	if n_samples == 1:
		mcmc_file = os.path.join(output_dir, f"{star_name}_mcmc.hdf5")
		samples_for_corner = run_mcmc(data, joker_samples, prior, joker, mcmc_file)
	else:
		samples_for_corner = joker_samples
	
	corner_plot(star_name, data, samples_for_corner, output_dir)












"""Fused GPU (Triton) recursions of the CaRT stores, each with an exact adjoint (hand-written or the generic stored-state adjoint).

They need a CUDA GPU.
  grstores   gprod_k (GR4J production store, percolation coefficient kq), gr_route_tv (GR4J routing store with exchange), quad_store
  pareto     pareto_prod (production store with Pareto-distributed capacities; also the depression store with net=False)
  top        top_layer (surface layer in series above the production store)
  zone       zone (near-stream saturated zone in parallel with the production store)
  theta      theta_state (connectivity wetness state)
  lin_store  lin_store (quick store with time-varying release coefficient)
  gw_et      gw_et (quadratic groundwater store with groundwater evapotranspiration)
  snow_sub   snow_bands_sub (elevation-band degree-day snowpack with sublimation)            -- CaRT, CaRT+ without P5
  snow_rp    snow_bands_rada, band_factors (snow_sub + additive PET-index melt term)      -- CaRT+
  adjoint    backward_k (generic stored-state adjoint used by several of the above)
"""

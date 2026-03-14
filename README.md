# postproc
Postprocessing CLI tools for use on SIGPROC filterbank files produced by Radio Telescopes (currently only validated for ATA data)

## Roadmap
- **Implement `rfiperf snr`**
  - parse `.bestprof`
  - compute SNR metrics
  - support comparison across N `.bestprof` files
  - overlay profile plots for visual comparison
  - JSON summary for compared inputs

- **Improve `rfiperf kurtosis` plot modes**
  - map antenna indices to real ATA antenna names
  - `ant` plots should label with antenna names
  - `freq` plots:
    - top axis = channel index
    - bottom axis = physical frequency units
  - `time` plots:
    - top axis = time-bin index
    - bottom axis = physical seconds
  - file output as the default, no plot popups

- **Make this a proper README**

- **Expand `rfiperf kurtosis` plot modes**
  - revisit Gurmehars tool for plotting methods
  - multi-antenna side-by-side views

- **Expand `rfiperf kurtosis` JSON modes**
  - save JSON to file by default
  - `-v` to also print to terminal
  - add `full`, for more info

- **Add `.fil` waterfall plotting**
  - simple saved PNGs

#### Nice to have if there is time
- **Add `.fil` handling**
  - `.fil` header parsing
  - support varying `nifs`
  - implement `postproc_common.filio`
  - implement `filsplice`
  - implement `readfil`

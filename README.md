[![Cite with Zenodo](http://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-1073c8?labelColor=000000)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![nf-test](https://img.shields.io/badge/unit_tests-nf--test-337ab7.svg)](https://www.nf-test.com)

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A524.04.2-23aa62.svg)](https://www.nextflow.io/)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?labelColor=000000&logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg?labelColor=000000)](https://sylabs.io/docs/)
[![Launch on Seqera Platform](https://img.shields.io/badge/Launch%20%F0%9F%9A%80-Seqera%20Platform-%234256e7)](https://cloud.seqera.io/launch?pipeline=https://github.com/open-bio/molflow)

## Introduction


```mermaid
flowchart TB
    subgraph MURUNDATA
    subgraph take
    v0["samples"]
    end
    v3([DATATETCHDRUG])
    v4([MUTILDATAPROCESS])
    v5([MUALLMODELS])
    v6([MUTRAIN])
    v7([VISIABLE])
    v8([SHAP])
    v0 --> v3
    v0 --> v4
    v3 --> v4
    v0 --> v5
    v4 --> v5
    v0 --> v6
    v5 --> v6
    v0 --> v7
    v6 --> v7
    v0 --> v8
    v3 --> v8
    v4 --> v8
    v5 --> v8
    v6 --> v8
    end
```


## Usage










## Credits




## Contributions and Support


## Citations





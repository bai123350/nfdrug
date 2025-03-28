include { DATATETCH               } from '../modules/local/datafetch/main'
include { DATAPROCESS             } from '../modules/local/dataprocess/main'
include { MODELS                  } from '../modules/local/models/main'
include { TRAIN                   } from '../modules/local/train/main'
include { DATATETCH } from '../modules/nf-core/datafetch/main.nf'



workflow RUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    // WORKFLOW: Run pipeline
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run DATATETCH
    DATATETCH (
        samplesheet
    )

    DATAPROCESS (
       samplesheet, DATATETCH.out.json
    )

    MODELS (
         samplesheet,DATAPROCESS.out.json, DATAPROCESS.out.npz
     )

    TRAIN (
        samplesheet,MODELS.out.pt, DATAPROCESS.out.npz
    )


    emit:
    json_report = DATATETCH.out.json // channel: /path/to/multiqc_report.html
}




include { DATATETCH               } from '../modules/nf-core/datafetch/main'
include { DATAPROCESS             } from '../modules/nf-core/dataprocess/main'
include { MODELS                  } from '../modules/nf-core/models/main.nf'
include { TRAIN                   } from '../modules/nf-core/train/main.nf'



workflow RUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    // WORKFLOW: Run pipeline
    // MOLFLOW (
    //     samplesheet
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run DATATETCH
    (meta_id,json) = DATATETCH (
        samplesheet
    )

    // println(res_json)
    // Output the results of DATATETCH
    // DATATETCH.out.json.view()

    // json.view()
    DATAPROCESS (
       samplesheet,json
    )


    MODELS (
         samplesheet,DATAPROCESS.out.json, DATAPROCESS.out.npz
     )

    TRAIN (
        samplesheet,DATAPROCESS.out.json, DATAPROCESS.out.npz
    )




    emit:
    json_report = DATATETCH.out.json // channel: /path/to/multiqc_report.html
}

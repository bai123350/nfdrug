


// include {} from '../modules/nf-core/models/main'

include { DATATETCH               } from '../modules/nf-core/datafetch/main'
include { DATAPROCESS             } from '../modules/nf-core/dataprocess/main'



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




    emit:
    json_report = DATATETCH.out.json // channel: /path/to/multiqc_report.html
}

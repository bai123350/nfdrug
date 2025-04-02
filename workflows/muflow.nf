include { DATATETCH   } from '../modules/local/datafetch/main'
include { MUTILDATAPROCESS } from '../modules/local/dataprocess/mudrug'
include { MUALLMODELS } from '../modules/local/models/mumodel'
include { MUTRAIN } from '../modules/local/train/mutrain'




workflow MURUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    // WORKFLOW: Run pipeline
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run DATATETCH
    // println(samplesheet.map { meta, protein1, protein2, _mo -> [meta, protein1, protein2] })
    DATATETCH(
        samplesheet
    )

    MUTILDATAPROCESS(
        samplesheet,
        DATATETCH.out.json,
    )

    MUALLMODELS(
        samplesheet,
        MUTILDATAPROCESS.out.all_folders
    )

    MUTRAIN(
        samplesheet,
        MUALLMODELS.out.model_dirs,
    )

    // emit:
    // json_report = DATATETCH.out.json // channel: /path/to/multiqc_report.html
}

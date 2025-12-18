const fs = require('fs')
const path = require('path')
const jsonbinpack = require('jsonbinpack')

excludedSchemas = new Set(["json-schema-for-codeship-pro-services-configuration-files.json", "mtad-yaml.json", "oss-review-toolkit-resolutions.json", "pyproject.json"])

// -------------------------
// CLI arguments
// -------------------------
const schemaDir = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve('./ReCG_schemas')

const documentsPath = process.argv[3]
if (!documentsPath) {
  console.error('Usage: node json_binpack.js <schema_dir> <documents.json>')
  process.exit(1)
}


// -------------------------
// Load JSONL documents from file
// -------------------------
const loadDocuments = (filePath) => {
  const documents = []
  const raw = fs.readFileSync(filePath, 'utf-8')
  const lines = raw.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue

    try {
      documents.push(JSON.parse(line))
    } catch (err) {
      console.error(`Invalid JSON at ${filePath}:${i + 1}`)
    }
  }

  return documents
}


// -------------------------
// Recursively load schemas
// -------------------------
const loadSchemas = (dir) => {
  const schemas = {}
  const files = fs.readdirSync(dir)

  for (const file of files) {
    const filePath = path.join(dir, file)
    const stat = fs.statSync(filePath)

    if (stat.isDirectory()) {
      Object.assign(schemas, loadSchemas(filePath))
    } 
    else if (file.endsWith('.json')) {
      try {
        const raw = fs.readFileSync(filePath, 'utf-8')
        schemas[filePath] = JSON.parse(raw)
      } catch (err) {
        console.error(`Failed to parse schema ${filePath}:`, err.message)
      }
    }
  }

  return schemas
}


// -------------------------
// Main logic
// -------------------------
const main = async () => {
  const schemas = loadSchemas(schemaDir)
  let totalDataSize = 0
  let totalSchemaSize = 0
  let averageSize = 0
  let schemaCount = 0
  let count = 0

  for (const [schemaPath, schema] of Object.entries(schemas)) {
    const fileName = path.basename(schemaPath)
    
    if (excludedSchemas.has(fileName)) {
      console.warn(`Skipping excluded schema: ${fileName}`)
      continue
    }

    const baseName = fileName.replace(/\.json$/, '')
    const datasetPath = path.join(documentsPath, `${baseName}.json`)

    if (!fs.existsSync(datasetPath)) {
      console.warn(`No dataset found for schema ${baseName}`)
      continue
    }

    const documents = loadDocuments(datasetPath)
    if (documents.length === 0) {
      console.warn(`No valid documents for ${baseName}`)
      continue
    }

    const currentSchemaSize = Buffer.byteLength(JSON.stringify(schema), 'utf-8')
    totalSchemaSize += currentSchemaSize

    try {
      const encodingSchema = await jsonbinpack.compileSchema(schema)
      schemaCount++

      for (const doc of documents) {
        try {
          const buffer = jsonbinpack.serialize(encodingSchema, doc)
          totalDataSize += buffer.length
          count++
        } catch (err) {
          // Document doesn't conform → skip
        }
      }

      if (count > 0) {
        const average = totalDataSize / count
        console.log(
          `Dataset: ${baseName} | Docs: ${count} | Avg size: ${average.toFixed(2)} bytes`
        )
        averageSize += average
      }
    } catch (err) {
      console.error(`Failed to process schema ${baseName}:`, err.message)
    }
  }

  if (schemaCount > 0) {
    console.log(`Overall average doc size: ${(averageSize / schemaCount).toFixed(2)} bytes`)
    console.log(`Overall average schema size: ${(totalSchemaSize / schemaCount).toFixed(2)} bytes`)
  } else {
    console.log('No schemas processed.')
  }
}


main().catch(err => {
  console.error(err)
  process.exit(1)
})

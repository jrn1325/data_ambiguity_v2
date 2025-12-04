const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const schemaDir = './processed_schemas'; // Input directory
const outputDir = './converted_processed_schemas'; // Output directory

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
}

// Function to determine the draft version from the $schema field
const getDraftVersion = (schema) => {
  const schemaUrl = schema.$schema;
  if (schemaUrl) {
    if (schemaUrl.includes('2020-12')) return '2020-12';
    if (schemaUrl.includes('2019-09')) return '2019-09';
    if (schemaUrl.includes('draft-07')) return 'draft7';
    if (schemaUrl.includes('draft-06')) return 'draft6';
    if (schemaUrl.includes('draft-04')) return 'draft4';
    if (schemaUrl.includes('draft-03')) return 'draft3';
  }
  return 'draft7'; // Default
};

// Convert schema or copy if already in 2020-12
const processSchema = (filePath) => {
  const raw = fs.readFileSync(filePath, 'utf-8');

  // Check for empty or invalid JSON
  if (!raw.trim()) {
    console.error(` Skipping empty file: ${filePath}`);
    return;
  }

  let schema;
  try {
    schema = JSON.parse(raw);
  } catch (err) {
    console.error(`Failed to parse JSON in ${filePath}: ${err.message}`);
    return;
  }

  const currentDraft = getDraftVersion(schema);
  const outputFilePath = path.join(outputDir, path.basename(filePath));

  // Skip if output file already exists (optional optimization)
  if (fs.existsSync(outputFilePath)) {
    console.log(`Skipping already converted file: ${outputFilePath}`);
    return;
  }

  if (currentDraft === '2020-12') {
    console.log(` ${filePath} is already draft-2020-12 → copying.`);
    fs.copyFileSync(filePath, outputFilePath);
    return;
  }

  if (currentDraft) {
    console.log(`Converting ${filePath} from ${currentDraft} → draft-2020-12...`);
    try {
      execSync(
        `alterschema --from ${currentDraft} --to 2020-12 "${filePath}" > "${outputFilePath}"`,
        { stdio: 'inherit' }
      );
      console.log(`Converted: ${filePath} → ${outputFilePath}`);
    } catch (error) {
      console.error(`Error converting ${filePath}: ${error.message}`);
    }
  } else {
    console.log(`Skipping ${filePath}, no recognized draft version found.`);
  }
};

// Process all JSON schema files in the directory
fs.readdirSync(schemaDir).forEach((fileName) => {
  const filePath = path.join(schemaDir, fileName);

  // Skip directories or non-JSON files
  if (!fs.lstatSync(filePath).isFile() || !fileName.endsWith('.json')) {
    console.log(`Skipping non-JSON or directory: ${filePath}`);
    return;
  }

  processSchema(filePath);
});

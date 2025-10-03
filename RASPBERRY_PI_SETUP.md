# Raspberry Pi Setup Guide

This guide explains how to clone only the rover folder on your Raspberry Pi to save space.

## Important: Existing Repository on Raspberry Pi

**Note:** If you already have the `quantum-tracer-il` repository on your Raspberry Pi, you have a few options:

### Option 1: Update the Existing Repository (Recommended)
Change the remote URL to point to the new Erewhon repository:

```bash
cd /path/to/quantum-tracer-il
git remote set-url origin https://github.com/maxboels/Erewhon.git
git fetch origin
git checkout -b main origin/main
```

### Option 2: Rename and Start Fresh
Rename the old repository and create a new sparse-checkout clone:

```bash
mv quantum-tracer-il quantum-tracer-il.backup
# Then follow the "Initial Setup on Raspberry Pi" instructions below
```

### Option 3: Remove and Clone Fresh
If you don't need the old repository anymore:

```bash
# Make sure you've pushed any important changes first!
rm -rf quantum-tracer-il
# Then follow the "Initial Setup on Raspberry Pi" instructions below
```

## Initial Setup on Raspberry Pi

When setting up on the Raspberry Pi for the first time, use sparse-checkout to clone only the rover folder:

```bash
# Clone the repository without checking out files
git clone --no-checkout https://github.com/maxboels/Erewhon.git
cd Erewhon

# Enable sparse-checkout
git sparse-checkout init --cone

# Set which folders to include (only rover)
git sparse-checkout set src/robots/rover

# Checkout the files
git checkout main
```

This will only download and track files in the `src/robots/rover` folder, saving significant disk space.

## Updating Git Remote on Raspberry Pi (One-Time Setup)

If you already have the old `quantum-tracer-il` repository on the Raspberry Pi, here's how to update it to point to the new Erewhon repository:

### Step 1: SSH into Raspberry Pi
```bash
ssh pi@<raspberry-pi-ip>
```

### Step 2: Navigate to the Repository
```bash
cd /path/to/quantum-tracer-il  # or wherever your repo is
```

### Step 3: Check Current Remote
```bash
git remote -v
# Should show: origin  https://github.com/maxboels/quantum-tracer-il.git
```

### Step 4: Update Remote URL
```bash
# Update the remote to point to Erewhon
git remote set-url origin https://github.com/maxboels/Erewhon.git

# Verify the change
git remote -v
# Should now show: origin  https://github.com/maxboels/Erewhon.git
```

### Step 5: Fetch and Switch to Main Branch
```bash
# Fetch the new repository structure
git fetch origin

# Switch to the main branch of Erewhon
git checkout main

# If you get an error, force the switch:
git checkout -B main origin/main
```

### Step 6: Set Up Sparse Checkout (Optional but Recommended)
Since the Raspberry Pi only needs the rover folder, configure sparse checkout:

```bash
# Enable sparse checkout
git config core.sparseCheckout true

# Specify only the rover folder
echo "src/robots/rover/" > .git/info/sparse-checkout

# Update the working directory
git checkout main
```

### Step 7: Verify Setup
```bash
# Check that only rover files are present
ls -la
# You should see only src/robots/rover/ and the root files

# Check git status
git status
# Should show "On branch main" with a clean working tree
```

## Daily Workflow on Raspberry Pi

Once set up, you can work normally:

```bash
# Pull latest changes (only rover folder if sparse checkout is enabled)
git pull

# Make your changes to files in src/robots/rover/

# Stage changes
git add .

# Commit
git commit -m "Your commit message"

# Push changes
git push
```

### Troubleshooting

**If git pull shows conflicts or unwanted files:**
```bash
# Reset to the remote state
git fetch origin
git reset --hard origin/main
```

**If you need to check what's being tracked:**
```bash
# See sparse checkout configuration
cat .git/info/sparse-checkout

# List all tracked files
git ls-files
```

**If you want to disable sparse checkout and see all files:**
```bash
git sparse-checkout disable
```

## Viewing What's Included in Sparse Checkout

```bash
# See what folders are currently included
git sparse-checkout list

# Add more folders if needed (e.g., to also include configs)
git sparse-checkout add src/configs
```

## Switching Back to Full Checkout (on laptop/desktop)

If you want to disable sparse-checkout and see all files:

```bash
git sparse-checkout disable
```

## Notes

- The sparse-checkout configuration is local to each clone
- Your laptop can have the full repository while the Raspberry Pi only has the rover folder
- Both can push and pull to the same remote repository
- Changes made in the rover folder on the Pi will sync with the full repository on your laptop

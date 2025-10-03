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

## Daily Workflow on Raspberry Pi

Once set up, you can work normally:

```bash
# Pull latest changes (only rover folder)
git pull

# Make your changes to files in src/robots/rover/

# Stage changes
git add .

# Commit
git commit -m "Your commit message"

# Push changes
git push
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

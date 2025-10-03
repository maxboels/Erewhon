# Erewhon Repository Setup Summary

## What Was Done

### 1. Cloned Imitation Learning from QuantumTracer
- **Source**: `https://github.com/maxboels/QuantumTracer.git` at path `/src/imitation_learning`
- **Destination**: `/home/maxboels/projects/Erewhon/src/policies/imitation_learning`
- **Method**: Sparse checkout to get only the imitation_learning folder
- **Status**: ✅ Complete

### 2. Cloned Quantum-Tracer-IL Contents to Rover
- **Source**: `https://github.com/maxboels/quantum-tracer-il.git` (entire repo contents)
- **Destination**: `/home/maxboels/projects/Erewhon/src/robots/rover`
- **Method**: Full clone, then moved contents (not the root folder itself)
- **Status**: ✅ Complete

### 3. Initialized Erewhon as Git Repository
- Created main branch
- Added `.gitignore` for Python projects
- Created `RASPBERRY_PI_SETUP.md` with sparse-checkout instructions
- **Status**: ✅ Complete

## Repository Structure

```
Erewhon/
├── .gitignore
├── README.md
├── RASPBERRY_PI_SETUP.md          # Instructions for Raspberry Pi setup
├── docs/
└── src/
    ├── cameras/
    ├── configs/
    ├── datasets/
    ├── policies/
    │   └── imitation_learning/    # From QuantumTracer repo
    ├── rl/
    ├── robots/
    │   ├── drone/
    │   └── rover/                 # From quantum-tracer-il repo
    ├── teleop/
    └── utils/
```

## Raspberry Pi Workflow

### Important: Transition from quantum-tracer-il

If your Raspberry Pi already has the `quantum-tracer-il` repository, you need to transition it to the new Erewhon repository. You have three options:

**Option 1 - Update Remote URL (Quickest)**
```bash
cd /path/to/quantum-tracer-il
git remote set-url origin https://github.com/YOUR_USERNAME/Erewhon.git
git fetch origin
git checkout -b main origin/main
```

**Option 2 - Rename and Fresh Clone**
```bash
mv quantum-tracer-il quantum-tracer-il.backup
# Then follow sparse checkout instructions below
```

**Option 3 - Remove and Fresh Clone**
```bash
rm -rf quantum-tracer-il  # Only if you've pushed all changes!
# Then follow sparse checkout instructions below
```

### On Raspberry Pi (Sparse Checkout)

When you SSH into the Raspberry Pi, clone only the rover folder:

```bash
git clone --no-checkout https://github.com/YOUR_USERNAME/Erewhon.git
cd Erewhon
git sparse-checkout init --cone
git sparse-checkout set src/robots/rover
git checkout main
```

This downloads ONLY the rover folder to save space on the embedded system.

### Daily Work on Raspberry Pi

```bash
cd Erewhon/src/robots/rover
# Make your changes
git add .
git commit -m "Your message"
git push
```

### On Your Laptop/Desktop (Full Checkout)

You have the complete repository with all folders. When you pull changes:

```bash
git pull
```

You'll get the rover updates along with everything else.

## Next Steps

1. **Create GitHub repository**: Create a new repository on GitHub called "Erewhon"
   
2. **Add remote and push**:
   ```bash
   cd /home/maxboels/projects/Erewhon
   git remote add origin https://github.com/YOUR_USERNAME/Erewhon.git
   git push -u origin main
   ```

3. **Set up Raspberry Pi**: Follow the instructions in `RASPBERRY_PI_SETUP.md`

## Key Benefits

- ✅ One unified repository for all Erewhon code
- ✅ Raspberry Pi only stores what it needs (rover folder)
- ✅ Laptop has full codebase
- ✅ Both can push/pull seamlessly
- ✅ No git submodules or complex configurations
- ✅ Uses standard Git sparse-checkout feature

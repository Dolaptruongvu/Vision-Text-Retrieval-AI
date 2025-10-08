// Quick MongoDB Connection Test
// Run: node test_connection.js

require('dotenv').config();
const mongoose = require('mongoose');

console.log('='.repeat(60));
console.log('MONGODB CONNECTION TEST');
console.log('='.repeat(60));
console.log(`Testing connection to: ${process.env.MONGODB_URI}`);
console.log('='.repeat(60));

mongoose.connect(process.env.MONGODB_URI)
  .then(() => {
    console.log('\n✅ SUCCESS! MongoDB Connected');
    console.log(`   Host: ${mongoose.connection.host}`);
    console.log(`   Port: ${mongoose.connection.port}`);
    console.log(`   Database: ${mongoose.connection.name}`);
    console.log(`   State: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Not Connected'}`);
    console.log('\n📊 Testing database operations...');
    
    // List collections
    return mongoose.connection.db.listCollections().toArray();
  })
  .then((collections) => {
    console.log(`\n📁 Existing collections in '${mongoose.connection.name}' database:`);
    if (collections.length === 0) {
      console.log('   (No collections yet - this is normal for a new database)');
    } else {
      collections.forEach((col) => {
        console.log(`   - ${col.name}`);
      });
    }
    console.log('\n' + '='.repeat(60));
    console.log('✅ ALL TESTS PASSED! MongoDB is ready to use.');
    console.log('='.repeat(60));
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n❌ CONNECTION FAILED!');
    console.error(`   Error: ${error.message}`);
    console.error('\n💡 Troubleshooting:');
    console.error('   1. Check if MongoDB is running:');
    console.error('      Windows: net start MongoDB');
    console.error('      Or manually: mongod --dbpath C:\\data\\db');
    console.error('   2. Verify port 27017 is available:');
    console.error('      netstat -ano | findstr :27017');
    console.error('   3. Check .env file has correct MONGODB_URI');
    console.error('   4. Try connecting with MongoDB Compass:');
    console.error('      mongodb://localhost:27017/plantAI');
    console.error('='.repeat(60));
    process.exit(1);
  });
